"""Record one five-life Breakout game using the trained agent."""
import argparse
import os

import imageio.v2 as imageio

from argument import add_arguments
from environment import Environment
from agent_dqn import Agent_DQN

EVAL_SEED = 11037


parser = argparse.ArgumentParser()
parser.add_argument('--video_folder', default='test_videos')
parser.add_argument('--episode_index', type=int, default=1,
                    help='1-based five-life evaluation episode to record')
parser = add_arguments(parser)
args = parser.parse_args()
args.test_dqn = True
args.train_dqn_again = False

os.makedirs(args.video_folder, exist_ok=True)
env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True,
                  test=True, render_mode='rgb_array')
agent = Agent_DQN(env, args)
env.reset(seed=EVAL_SEED)

total_reward = 0.0
video_path = os.path.join(args.video_folder, 'breakout_combined.mp4')
writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)

def play_one_game(record=False):
    total = 0.0
    frames = []
    for _ in range(5):
        state = env.reset()
        agent.init_game_setting()
        terminated = truncated = False
        if record:
            frames.append(env.env.render())
        while not terminated and not truncated:
            previous_state = state
            action = agent.make_action(state, test=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            observe = getattr(agent, 'observe_transition', None)
            if observe is not None:
                observe(previous_state, action, reward, state,
                        terminated, truncated)
            if record:
                frames.append(env.env.render())
    return total, frames

# Replay earlier five-life games without recording so the requested game is
# aligned with the same deterministic evaluation sequence.
for _ in range(max(0, args.episode_index - 1)):
    play_one_game(record=False)

total_reward, frames = play_one_game(record=True)
for frame in frames:
    writer.append_data(frame)

writer.close()
env.close()
print(f'Recorded episode {args.episode_index}: {video_path}; five-life reward: {total_reward}')
