# Breakout DQN experiment log

This is the working record for the Project 3 report. Scores are game rewards
over five lives per evaluation episode. The course submission must leave
`test.py`, `environment.py`, `atari_wrapper.py`, `main.py`, and `agent.py`
unchanged; model changes belong in `agent_dqn.py` and `dqn_model.py`.

## Confirmed experiments

| ID | Model / setup | Training or evaluation setup | Result | Decision |
| --- | --- | --- | --- | --- |
| E1 | `per_finetune_10m.pth` | 100 evaluation episodes | Mean **346.88**; 42:18 evaluation time | Strong trained checkpoint; retained. |
| E2 | Evaluation escape maneuver | A deterministic paddle sweep after a long interval with no positive reward | 100-episode mean **400.76** in the recorded run | Improved that recorded evaluation, but the video still showed repeated late-game trajectories. It is disabled by default because it is a brittle evaluation heuristic, not a learned improvement. |
| E3 | `safe_gpu0_final.pth` | 20 diagnostic evaluation episodes, no scripted escape | Mean **360.55** | Current protected learned baseline. Do not overwrite. |
| E4 | `dqn_3step_5m.pth` | Fresh 5M-step Dueling Double DQN + PER + 3-step return run; server restart resumed from the 4.75M checkpoint | 20-episode mean **35.30** | Rejected. The fresh 3-step training configuration did not learn a useful policy. |
| E5 | `safe_gpu0_3step_finetune_1m.pth` | Fine-tuned E3 weights for 1M steps with PER, 3-step returns, lr `2.5e-5`, epsilon `0.15 -> 0.02` | 20-episode mean **149.25** | Rejected. Adding 3-step targets to this baseline degraded the policy. |

### Interpretation

The baseline already far exceeds the 40-point mean required for full model
credit. The two 3-step candidates performed worse, so no 3-step checkpoint
will be used for submission. The observed leaderboard score of 723 came from
systematic CNN/Dueling-DQN tuning; the course also links the Rainbow DQN paper.

`E2` should not be treated as a controlled comparison against `E1`: the
terminal capture did not preserve the exact checkpoint name for every earlier
run. The reproducible learned baseline is `safe_gpu0_final.pth` in E3.

## Next experiment plan

### P1 — Rainbow-lite leaderboard candidate

Implement and train one new candidate entirely through the allowed
`agent_dqn.py` and `dqn_model.py` files. Keep the working Nature-DQN CNN,
Dueling head, Double-DQN target, replay, target network, Huber loss, and
prioritized replay. Add NoisyNet exploration and C51 distributional value
learning. Do **not** include 3-step returns in the first version because E4
and E5 rejected that component under our current setup.

First run a 25K-step smoke test in Jupyter to verify the C51 loss and CUDA
environment. Then train for up to 5M steps, preserve checkpoints every 250K
steps, and evaluate every promising checkpoint with the unchanged 100-episode
course evaluator. Retain the highest-scoring model, not merely the final one.

### Evaluation rule

Use the same 100-episode evaluator and the same five-life setup for every
leaderboard comparison. Twenty-episode runs are diagnostics only; they do not
replace the official result.

### P1a — runtime smoke test

Completed successfully on CUDA with two workers: `rainbow_train.py` reached
25K steps, produced finite C51 losses (3.9322 at 20K and 3.0744 at 25K), and
saved both numbered and rolling checkpoints. This confirms that the NoisyNet,
C51 projection, prioritized replay, and vector Atari pipeline execute together.

### P2 — controlled propagation experiments

The next experiments isolate two plausible credit-assignment mechanisms from
the 5M Rainbow baseline:

| Arm | Only changed setting | Duration | Decision criterion |
| --- | --- | --- | --- |
| P2-A | `n_step=3` rather than `n_step=1` | 1.5M steps | At least 50 mean above the matched 1-step checkpoint at two consecutive comparisons, without increasing the rate of scores below 100. |
| P2-B | Target update every 2,000 gradients rather than 8,000 | 1.5M steps | Same criterion, evaluated independently. |

`rainbow_utils.NStepAccumulator` is unit-tested for full windows, terminal
flushes, and independent worker trajectories. The C51 projection uses
`gamma ** horizon`, masks bootstrapping at terminal/life-loss transitions, and
stores the state reached after the full n-step window.
