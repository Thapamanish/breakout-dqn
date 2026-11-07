#!/usr/bin/env python3
"""Git-synced autonomous evaluation & experiment runner for JupyterHub GPUs.

Usage on JupyterHub (in ~/RL/breakout-dqn):
    git pull && nohup python3 auto_runner.py > logs/auto_runner.log 2>&1 &

What this daemon does:
1. Snapshots workspace status (GPUs, running processes, *.pth files, existing
   proxy_curves.txt and logs/*.log tails) into results/ and pushes to GitHub.
2. Automatically screens unscreened checkpoints (including
   rainbow_target2k_5m_2750k..5000k and rainbow_n3_t2k_1500k_250k..1500k, plus
   any newly produced *_<step>k.pth checkpoints) with eval_proxy.py.
3. Automatically runs official 100-episode evaluation (main.py --test_dqn) on
   the highest-proxy checkpoint exceeding the current official threshold
   (277.16), saving full output to results/official_scores.json.
4. Polls results/job_queue.json via `git pull` every 45 seconds so new
   training/evaluation jobs pushed to GitHub execute automatically.

Protected course files (test.py, environment.py, atari_wrapper.py, main.py,
agent.py, agent_dqn.py) and *.pth checkpoints are NEVER modified.
"""

import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_DIR, "results")
JOB_OUTPUTS_DIR = os.path.join(RESULTS_DIR, "job_outputs")
PROXY_JSON = os.path.join(RESULTS_DIR, "proxy_scores.json")
OFFICIAL_JSON = os.path.join(RESULTS_DIR, "official_scores.json")
STATE_JSON = os.path.join(RESULTS_DIR, "runner_state.json")
SUMMARY_MD = os.path.join(RESULTS_DIR, "summary.md")
SNAPSHOT_TXT = os.path.join(RESULTS_DIR, "system_snapshot.txt")
JOB_QUEUE_JSON = os.path.join(RESULTS_DIR, "job_queue.json")

BASELINE_OFFICIAL_THRESHOLD = 277.16

KNOWN_PROXY_SCORES = {
    "rainbow_c51_noisy_5m_250k.pth": {"proxy": 2.40},
    "rainbow_c51_noisy_5m_500k.pth": {"proxy": 3.06},
    "rainbow_c51_noisy_5m_750k.pth": {"proxy": 0.60},
    "rainbow_c51_noisy_5m_1000k.pth": {"proxy": 5.40},
    "rainbow_c51_noisy_5m_1250k.pth": {"proxy": 25.28},
    "rainbow_c51_noisy_5m_1500k.pth": {"proxy": 37.58},
    "rainbow_n3_1500k_250k.pth": {"proxy": 2.14},
    "rainbow_n3_1500k_500k.pth": {"proxy": 23.56},
    "rainbow_n3_1500k_750k.pth": {"proxy": 36.76},
    "rainbow_n3_1500k_1000k.pth": {"proxy": 50.94},
    "rainbow_n3_1500k_1250k.pth": {"proxy": 64.26},
    "rainbow_n3_1500k_1500k.pth": {"proxy": 71.96},
    "rainbow_target2k_1500k_250k.pth": {"proxy": 6.00},
    "rainbow_target2k_1500k_500k.pth": {"proxy": 13.16},
    "rainbow_target2k_1500k_750k.pth": {"proxy": 20.52},
    "rainbow_target2k_1500k_1000k.pth": {"proxy": 36.22},
    "rainbow_target2k_1500k_1250k.pth": {"proxy": 54.06},
    "rainbow_target2k_1500k_1500k.pth": {"proxy": 147.98},
    "rainbow_target2k_5m_1750k.pth": {"proxy": 176.18},
    "rainbow_target2k_5m_2000k.pth": {"proxy": 217.76},
    "rainbow_target2k_5m_2250k.pth": {"proxy": 133.28},
    "rainbow_target2k_5m_2500k.pth": {"proxy": 277.16},
}

KNOWN_OFFICIAL_SCORES = {
    "rainbow_c51_noisy_5m.pth": {"mean": 359.50, "note": "5M baseline (target update 8000)"},
    "per_finetune_10m.pth": {"mean": 346.88, "note": "Earlier PER 10M"},
    "rainbow_c51_noisy_5m_1250k.pth": {"mean": 25.28},
    "rainbow_c51_noisy_5m_1500k.pth": {"mean": 37.58},
    "rainbow_n3_1500k_1250k.pth": {"mean": 64.27},
    "rainbow_n3_1500k_1500k.pth": {"mean": 71.96},
    "rainbow_target2k_1500k_1250k.pth": {"mean": 54.06},
    "rainbow_target2k_1500k_1500k.pth": {"mean": 147.98},
    "rainbow_target2k_5m_1750k.pth": {"mean": 176.18},
    "rainbow_target2k_5m_2000k.pth": {"mean": 217.76},
    "rainbow_target2k_5m_2500k.pth": {"mean": 277.16},
}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{now_iso()}] {msg}", flush=True)


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            log(f"Warning: could not parse {path}: {exc}")
    return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def run_cmd(cmd, cwd=REPO_DIR, env=None, timeout=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def pick_best_gpu():
    """Return the GPU index with the lowest memory usage, or None if CPU-only."""
    try:
        res = run_cmd(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return None
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                gpus.append((int(parts[1]), parts[0]))
        if gpus:
            gpus.sort()
            return gpus[0][1]
    except Exception:
        pass
    return None


def is_checkpoint_ready(path):
    """Verify a .pth checkpoint is completely written and loadable."""
    if not os.path.exists(path):
        return False
    try:
        stat = os.stat(path)
        if stat.st_size < 1_000_000:
            return False
        if time.time() - stat.st_mtime < 8.0:
            return False
        import torch
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "online_net" not in ckpt:
            return False
        return True
    except Exception as exc:
        log(f"Checkpoint {path} not ready yet: {exc}")
        return False


def write_summary(proxies, officials, state):
    lines = [
        "# Automated Experiment & Evaluation Summary",
        "",
        f"Last updated: `{now_iso()}`  ",
        f"Runner status: `{state.get('status', 'running')}`  ",
        f"Current task: `{state.get('current_task', 'idle')}`",
        "",
        "## Official 100-Episode Evaluations",
        "",
        "| Checkpoint | Official Mean | Proxy | Notes |",
        "| --- | ---: | ---: | --- |",
    ]
    sorted_officials = sorted(
        officials.items(),
        key=lambda kv: float(kv[1].get("mean", -1.0)),
        reverse=True,
    )
    for ckpt, info in sorted_officials:
        mean_val = info.get("mean")
        mean_str = f"{mean_val:.2f}" if isinstance(mean_val, (int, float)) else str(mean_val)
        proxy_val = proxies.get(ckpt, {}).get("proxy")
        proxy_str = f"{proxy_val:.2f}" if isinstance(proxy_val, (int, float)) else "-"
        note = info.get("note", "")
        lines.append(f"| `{ckpt}` | **{mean_str}** | {proxy_str} | {note} |")

    lines.extend([
        "",
        "## Proxy Screening Scores (First 9 Fixed-Seed Episodes)",
        "",
        "| Checkpoint | Proxy Score | Episode Scores | Evaluated At |",
        "| --- | ---: | --- | --- |",
    ])

    def ckpt_sort_key(item):
        name = item[0]
        m = re.search(r"_(\d+)k\.pth$", name)
        step = int(m.group(1)) if m else 999999
        prefix = re.sub(r"_\d+k\.pth$", "", name)
        return (prefix, step)

    for ckpt, info in sorted(proxies.items(), key=ckpt_sort_key):
        proxy_val = info.get("proxy")
        proxy_str = f"{proxy_val:.2f}" if isinstance(proxy_val, (int, float)) else str(proxy_val)
        scores_str = str(info.get("scores", "-"))
        ts = info.get("evaluated_at", "-")
        lines.append(f"| `{ckpt}` | **{proxy_str}** | `{scores_str}` | {ts} |")

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def capture_snapshot():
    sections = [f"Snapshot time: {now_iso()}\n"]

    # GPU status
    res = run_cmd(["nvidia-smi"], timeout=15)
    sections.append("=== nvidia-smi ===\n" + (res.stdout if res.returncode == 0 else "nvidia-smi unavailable\n"))

    # Running python processes
    res = run_cmd(["bash", "-c", "ps -ef | grep '[p]ython' || true"], timeout=10)
    sections.append("=== Running Python Processes ===\n" + res.stdout)

    # Checkpoints
    res = run_cmd(["bash", "-c", "ls -lh *.pth 2>/dev/null || true"], timeout=10)
    sections.append("=== Checkpoints (*.pth) ===\n" + res.stdout)

    # Existing text files (proxy_curves.txt, proxy_*.txt)
    for txt_path in sorted(glob.glob(os.path.join(REPO_DIR, "proxy*.txt"))):
        name = os.path.basename(txt_path)
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()[-12000:]
            sections.append(f"=== {name} ===\n{content}\n")
        except Exception as exc:
            sections.append(f"=== {name} (error: {exc}) ===\n")

    # Tails of logs/*.log
    for log_path in sorted(glob.glob(os.path.join(REPO_DIR, "logs", "*.log"))):
        name = os.path.basename(log_path)
        if name == "auto_runner.log":
            continue
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-60:]
            sections.append(f"=== logs/{name} (last 60 lines) ===\n" + "".join(lines) + "\n")
        except Exception as exc:
            sections.append(f"=== logs/{name} (error: {exc}) ===\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SNAPSHOT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))


def git_sync(commit_message):
    """Stage results/, commit if changed, pull --rebase, and push to origin."""
    try:
        run_cmd(["git", "add", "-f", "results/"], timeout=20)
        status = run_cmd(["git", "status", "--porcelain", "results/"], timeout=20)
        if status.stdout.strip():
            commit_res = run_cmd(
                [
                    "git",
                    "-c", "user.name=auto-runner",
                    "-c", "user.email=mthapa2955@sdsu.edu",
                    "commit",
                    "-m", commit_message,
                    "--", "results/",
                ],
                timeout=30,
            )
            if commit_res.returncode != 0:
                log(f"git commit warning: {commit_res.stdout}")
        pull_res = run_cmd(["git", "pull", "--rebase", "--autostash", "origin", "main"], timeout=60)
        if pull_res.returncode != 0:
            log(f"git pull warning: {pull_res.stdout}")
        push_res = run_cmd(["git", "push", "origin", "main"], timeout=60)
        if push_res.returncode != 0:
            log(f"git push warning: {push_res.stdout}")
        else:
            log(f"Synced with GitHub: {commit_message}")
    except Exception as exc:
        log(f"git_sync error: {exc}")


def run_proxy_eval(ckpt_name, proxies, officials, state):
    ckpt_path = os.path.join(REPO_DIR, ckpt_name)
    if not is_checkpoint_ready(ckpt_path):
        return False

    gpu = os.environ.get("EVAL_GPU") or pick_best_gpu()
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {}
    state["current_task"] = f"eval_proxy on {ckpt_name} (GPU {gpu})"
    save_json(STATE_JSON, state)
    log(f"Running eval_proxy.py on {ckpt_name} (GPU={gpu})...")

    res = run_cmd(
        [sys.executable, "eval_proxy.py", "--model_path", ckpt_name],
        env=env,
        timeout=600,
    )
    output = res.stdout
    scores_match = re.search(r"scores\s+(\[[^\]]+\])", output)
    proxy_match = re.search(r"proxy:\s*([0-9.]+)", output)
    time_match = re.search(r"running time:\s*([0-9.]+)", output)

    if proxy_match:
        proxy_val = float(proxy_match.group(1))
        scores_list = json.loads(scores_match.group(1)) if scores_match else []
        elapsed = float(time_match.group(1)) if time_match else None
        proxies[ckpt_name] = {
            "proxy": proxy_val,
            "scores": scores_list,
            "elapsed_sec": round(elapsed, 2) if elapsed is not None else None,
            "evaluated_at": now_iso(),
        }
        save_json(PROXY_JSON, proxies)
        write_summary(proxies, officials, state)
        log(f"Proxy result for {ckpt_name}: proxy={proxy_val:.2f} scores={scores_list}")
        return True
    else:
        log(f"eval_proxy failed for {ckpt_name}:\n{output[-1000:]}")
        return False


def run_official_eval(ckpt_name, proxies, officials, state, note=""):
    ckpt_path = os.path.join(REPO_DIR, ckpt_name)
    if not is_checkpoint_ready(ckpt_path):
        log(f"Cannot run official eval; checkpoint not ready: {ckpt_name}")
        return False

    gpu = os.environ.get("EVAL_GPU") or pick_best_gpu()
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {}
    state["current_task"] = f"official 100-ep eval on {ckpt_name} (GPU {gpu})"
    save_json(STATE_JSON, state)
    write_summary(proxies, officials, state)
    git_sync(f"Start official 100-ep evaluation on {ckpt_name}")

    log(f"Running official 100-episode evaluation on {ckpt_name} (GPU={gpu})...")
    start_t = time.time()
    res = run_cmd(
        [sys.executable, "main.py", "--test_dqn", "--model_path", ckpt_name],
        env=env,
        timeout=7200,
    )
    elapsed = time.time() - start_t
    output = res.stdout

    os.makedirs(JOB_OUTPUTS_DIR, exist_ok=True)
    raw_log_path = os.path.join(JOB_OUTPUTS_DIR, f"official_{ckpt_name}.txt")
    with open(raw_log_path, "w", encoding="utf-8") as f:
        f.write(output)

    mean_match = re.search(r"Mean:\s*([0-9.]+)", output)
    if mean_match:
        mean_val = float(mean_match.group(1))
        officials[ckpt_name] = {
            "mean": mean_val,
            "elapsed_sec": round(elapsed, 2),
            "evaluated_at": now_iso(),
            "note": note or "Official 100-episode evaluation",
        }
        save_json(OFFICIAL_JSON, officials)
        state["current_task"] = "idle"
        save_json(STATE_JSON, state)
        write_summary(proxies, officials, state)
        git_sync(f"Official 100-ep result for {ckpt_name}: {mean_val:.2f}")
        log(f"Official 100-ep mean for {ckpt_name}: {mean_val:.2f}")
        return True
    else:
        log(f"Official evaluation failed for {ckpt_name}:\n{output[-1500:]}")
        state["current_task"] = f"failed official eval on {ckpt_name}"
        save_json(STATE_JSON, state)
        git_sync(f"Official eval failed log for {ckpt_name}")
        return False


def discover_unscreened_checkpoints(proxies):
    """Find priority checkpoints and any other *_<step>k.pth checkpoints."""
    ordered = []
    # Priority 1: target2k 5m continuation (2750k..5000k) and final
    for s in [2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000]:
        ordered.append(f"rainbow_target2k_5m_{s}k.pth")
    # Priority 2: n3_t2k_1500k arm (250k..1500k)
    for s in [250, 500, 750, 1000, 1250, 1500]:
        ordered.append(f"rainbow_n3_t2k_1500k_{s}k.pth")
    # Priority 3: any other numbered rainbow checkpoints in repo root (excluding old baseline/n3 arms)
    for path in sorted(glob.glob(os.path.join(REPO_DIR, "rainbow_*_*k.pth"))):
        name = os.path.basename(path)
        if name.startswith("rainbow_c51_noisy_5m_") or name.startswith("rainbow_n3_1500k_"):
            continue
        if name not in ordered:
            ordered.append(name)

    todo = []
    for name in ordered:
        if name not in proxies and is_checkpoint_ready(os.path.join(REPO_DIR, name)):
            todo.append(name)
    return todo


def process_job_queue(proxies, officials, state):
    queue_data = load_json(JOB_QUEUE_JSON, {"jobs": []})
    jobs = queue_data.get("jobs", [])
    completed = set(state.get("completed_jobs", []))
    did_work = False

    for job in jobs:
        job_id = job.get("id")
        if not job_id or job_id in completed:
            continue

        jtype = job.get("type")
        log(f"Executing queued job {job_id} (type={jtype})...")
        state["current_task"] = f"job {job_id}: {jtype}"
        save_json(STATE_JSON, state)

        if jtype == "proxy_eval":
            for ckpt in job.get("checkpoints", []):
                run_proxy_eval(ckpt, proxies, officials, state)
        elif jtype == "official_eval":
            ckpt = job.get("checkpoint")
            if ckpt:
                run_official_eval(ckpt, proxies, officials, state, note=job.get("note", ""))
        elif jtype == "train_bg":
            cmd = job.get("cmd")
            log_file = job.get("log_file", f"logs/{job_id}.log")
            gpu = job.get("gpu")
            if gpu is None:
                gpu = pick_best_gpu()
            os.makedirs(os.path.join(REPO_DIR, "logs"), exist_ok=True)
            log_abs = os.path.join(REPO_DIR, log_file)
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with open(log_abs, "a", encoding="utf-8") as lf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_DIR,
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            state.setdefault("spawned_pids", {})[job_id] = {
                "pid": proc.pid,
                "cmd": cmd,
                "gpu": gpu,
                "log_file": log_file,
                "started_at": now_iso(),
            }
            log(f"Started background training job {job_id} with PID {proc.pid} on GPU {gpu}")
        elif jtype == "shell":
            cmd = job.get("cmd", [])
            res = run_cmd(cmd, timeout=job.get("timeout", 300))
            os.makedirs(JOB_OUTPUTS_DIR, exist_ok=True)
            out_path = os.path.join(JOB_OUTPUTS_DIR, f"{job_id}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(res.stdout)

        completed.add(job_id)
        state["completed_jobs"] = sorted(completed)
        state["current_task"] = "idle"
        save_json(STATE_JSON, state)
        capture_snapshot()
        write_summary(proxies, officials, state)
        git_sync(f"Completed queued job {job_id}")
        did_work = True

    return did_work


def import_existing_proxy_files(proxies):
    """Parse any existing proxy*.txt files so already-screened checkpoints aren't re-run."""
    pattern = re.compile(
        r"==\s*([A-Za-z0-9_.-]+?)(?:\.pth)?\s*\n"
        r"(?:scores\s+(\[[^\]]+\])\s*\n)?"
        r"proxy:\s*([0-9.]+)",
        re.MULTILINE,
    )
    imported = 0
    for txt_path in sorted(glob.glob(os.path.join(REPO_DIR, "proxy*.txt"))):
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            for m in pattern.finditer(text):
                ckpt_base = m.group(1)
                ckpt_name = ckpt_base if ckpt_base.endswith(".pth") else f"{ckpt_base}.pth"
                if ckpt_name not in proxies:
                    scores_list = json.loads(m.group(2)) if m.group(2) else []
                    proxies[ckpt_name] = {
                        "proxy": float(m.group(3)),
                        "scores": scores_list,
                        "evaluated_at": "imported_from_txt",
                    }
                    imported += 1
        except Exception as exc:
            log(f"Warning parsing {txt_path}: {exc}")
    if imported:
        log(f"Imported {imported} proxy results from existing proxy*.txt files")
    return imported


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(JOB_OUTPUTS_DIR, exist_ok=True)

    proxies = load_json(PROXY_JSON, {})
    for k, v in KNOWN_PROXY_SCORES.items():
        proxies.setdefault(k, v)
    import_existing_proxy_files(proxies)
    save_json(PROXY_JSON, proxies)

    officials = load_json(OFFICIAL_JSON, {})
    for k, v in KNOWN_OFFICIAL_SCORES.items():
        officials.setdefault(k, v)
    save_json(OFFICIAL_JSON, officials)

    state = load_json(
        STATE_JSON,
        {
            "status": "running",
            "started_at": now_iso(),
            "current_task": "startup_snapshot",
            "completed_jobs": [],
        },
    )
    state["status"] = "running"
    state["last_heartbeat"] = now_iso()

    # Step 1: Immediate snapshot and push
    log("Capturing initial workspace snapshot...")
    capture_snapshot()
    write_summary(proxies, officials, state)
    save_json(STATE_JSON, state)
    git_sync("Initial workspace snapshot from JupyterHub auto_runner")
    last_snapshot_push = time.time()

    # Main autonomous loop
    while True:
        try:
            did_work = False

            # 1. Screen any unscreened checkpoints with eval_proxy.py
            todo = discover_unscreened_checkpoints(proxies)
            if todo:
                log(f"Found {len(todo)} unscreened checkpoints: {todo}")
                completed_since_sync = 0
                for ckpt in todo:
                    if run_proxy_eval(ckpt, proxies, officials, state):
                        completed_since_sync += 1
                        did_work = True
                    if completed_since_sync >= 2:
                        capture_snapshot()
                        git_sync(f"Update proxy screening ({ckpt})")
                        last_snapshot_push = time.time()
                        completed_since_sync = 0
                if completed_since_sync > 0:
                    capture_snapshot()
                    git_sync("Complete batch of proxy screenings")
                    last_snapshot_push = time.time()

            # 2. Check if any target2k / n3_t2k checkpoint beats the best official target2k score
            best_target2k_official = max(
                [BASELINE_OFFICIAL_THRESHOLD]
                + [
                    float(info.get("mean", 0.0))
                    for ckpt, info in officials.items()
                    if ckpt.startswith("rainbow_target2k") or ckpt.startswith("rainbow_n3_t2k")
                ]
            )
            candidates = [
                (float(info.get("proxy", 0.0)), ckpt)
                for ckpt, info in proxies.items()
                if (ckpt.startswith("rainbow_target2k_5m_") or ckpt.startswith("rainbow_n3_t2k_"))
                and ckpt not in officials
                and float(info.get("proxy", 0.0)) > best_target2k_official
            ]
            if candidates:
                candidates.sort(reverse=True)
                best_proxy, best_ckpt = candidates[0]
                log(
                    f"Top candidate {best_ckpt} has proxy {best_proxy:.2f} > {best_target2k_official:.2f}; "
                    f"launching official 100-episode evaluation!"
                )
                if run_official_eval(
                    best_ckpt,
                    proxies,
                    officials,
                    state,
                    note=f"Auto-triggered (proxy {best_proxy:.2f} > {best_target2k_official:.2f})",
                ):
                    did_work = True
                    last_snapshot_push = time.time()

            # 3. Pull latest job_queue.json from GitHub and execute any queued jobs
            run_cmd(["git", "pull", "--rebase", "--autostash", "origin", "main"], timeout=60)
            if process_job_queue(proxies, officials, state):
                did_work = True
                last_snapshot_push = time.time()

            # 4. If work was done or 10 minutes elapsed, refresh snapshot & push
            if did_work or (time.time() - last_snapshot_push >= 600):
                state["current_task"] = "idle"
                state["last_heartbeat"] = now_iso()
                save_json(STATE_JSON, state)
                capture_snapshot()
                write_summary(proxies, officials, state)
                git_sync("Refresh workspace snapshot")
                last_snapshot_push = time.time()

        except Exception as exc:
            log(f"Loop error: {exc}")

        time.sleep(45)


if __name__ == "__main__":
    main()
