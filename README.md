# Breakout DQN

Deep Q-learning experiments for Atari Breakout using Gymnasium and PyTorch.

## Project layout

- `main.py` — entry point for training and evaluation
- `agent_dqn.py` — DQN agent and replay buffer
- `dqn_model.py` — online and target network definitions
- `parallel_train.py` — vectorized training with optional prioritized replay
- `atari_wrapper.py` — Atari preprocessing and frame stacking
- `environment.py` — project environment adapter
- `test.py` — evaluation loop
- `record_preview.py` — video-preview helper
- `docs/` — tutorial notebook

Model checkpoints, videos, logs, caches, and virtual environments are intentionally excluded from Git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch separately for the target machine (CPU, CUDA, or Apple Silicon).

## Run evaluation

```bash
python3 main.py --test_dqn --model_path dqn_model.pth
```

To record evaluation videos:

```bash
python3 main.py --test_dqn --model_path dqn_model.pth --record_video
```

## Run parallel training

```bash
python3 parallel_train.py \
  --workers 2 \
  --gpu_count 1 \
  --resume dqn_model_baseline_345.pth \
  --output parallel_5m.pth
```

Boolean options such as `--prioritized_replay` are flags: use the option by itself, without `True` or `False`.

## Notes

The Atari environment is `BreakoutNoFrameskip-v4`. Training uses clipped rewards and evaluation uses the saved checkpoint specified by `--model_path`.
