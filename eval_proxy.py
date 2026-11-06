"""Fast, non-official fixed-seed proxy for the course Breakout evaluation.

This intentionally mirrors the evaluation path in ``main.py --test_dqn`` and
the first nine outer iterations in ``test.test``.  It is not a replacement for
the 100-episode course evaluation.
"""

import time

from main import parse as parse_official_args
from environment import Environment
from agent_dqn import Agent_DQN


OFFICIAL_SEED = 11037


def main():
    # Reuse main.py's parser so Agent_DQN receives precisely the same defaults.
    args = parse_official_args()
    # Agent_DQN uses this flag to require and load --model_path, as main.py does.
    args.test_dqn = True
    args.train_dqn = False
    args.train_dqn_again = False

    start_time = time.time()
    env = Environment("BreakoutNoFrameskip-v4", args, atari_wrapper=True, test=True)
    agent = Agent_DQN(env, args)

    scores = []
    env.reset(seed=OFFICIAL_SEED)
    truncated = False
    try:
        for _ in range(9):
            episode_reward = 0.0
            for _ in range(5):
                state = env.reset()
                agent.init_game_setting()
                terminated = False
                while not terminated and not truncated:
                    action = agent.make_action(state, test=True)
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                if truncated:
                    break
            scores.append(episode_reward)
    finally:
        env.close()

    proxy = (34 * scores[1] + 28 * scores[8] + 20 * scores[5] + 18 * scores[4]) / 100
    print("scores", scores)
    print(f"proxy: {proxy:.2f}")
    print(f"running time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
