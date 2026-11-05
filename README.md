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
  --prioritized_replay \
  --n_step 3 \
  --output dqn_3step_5m.pth
```

Boolean options such as `--prioritized_replay` are flags: use the option by itself, without `True` or `False`.

This command starts a clean run. To continue an existing training checkpoint,
add `--resume path/to/checkpoint.pth` explicitly.

To test a training improvement from an existing strong model without altering
it, use a fresh fine-tune. This resets the replay buffer, optimizer, target
network, and exploration schedule while retaining the model weights:

```bash
python3 parallel_train.py \
  --workers 2 \
  --gpu_count 1 \
  --max_steps 1000000 \
  --resume ~/RL/Project3_new/safe_gpu0_final.pth \
  --fresh_finetune \
  --prioritized_replay \
  --n_step 3 \
  --learning_rate 2.5e-5 \
  --epsilon_start 0.15 \
  --epsilon_end 0.02 \
  --epsilon_decay_steps 500000 \
  --output safe_gpu0_3step_finetune_1m.pth
```

## Run the Rainbow-lite leaderboard experiment

The Rainbow-lite trainer uses the existing Nature-DQN CNN with a dueling C51
distributional head, Double-DQN action selection, prioritized replay, and
NoisyNet exploration. It saves a rolling checkpoint and a separate checkpoint
every 250K environment steps.

```bash
python3 rainbow_train.py \
  --workers 2 \
  --max_steps 5000000 \
  --output rainbow_c51_noisy_5m.pth
```

If the Jupyter server restarts, resume from the most recent numbered
checkpoint. The replay buffer is rebuilt with 20K new transitions before the
next gradient update:

```bash
python3 rainbow_train.py \
  --workers 2 \
  --max_steps 5000000 \
  --resume rainbow_c51_noisy_2500k.pth \
  --output rainbow_c51_noisy_5m.pth
```

Evaluate any resulting checkpoint with the normal course entry point:

```bash
python3 main.py --test_dqn --model_path rainbow_c51_noisy_5m.pth
```

## Notes

The Atari environment is `BreakoutNoFrameskip-v4`. Training uses clipped rewards and evaluation uses the saved checkpoint specified by `--model_path`.
