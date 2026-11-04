"""Record one five-life Breakout game using the trained agent."""
import argparse
import os

import imageio.v2 as imageio

from argument import add_arguments
from environment import Environment
from agent_dqn import Agent_DQN


parser = argparse.ArgumentParser()
parser.add_argument('--video_folder', default='test_videos')
parser = add_arguments(parser)
args = parser.parse_args()
args.test_dqn = True
args.train_dqn_again = False

os.makedirs(args.video_folder, exist_ok=True)
env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True,
                  test=True, render_mode='rgb_array')
agent = Agent_DQN(env, args)

total_reward = 0.0
video_path = os.path.join(args.video_folder, 'breakout_combined.mp4')
writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)
for _ in range(5):
    state = env.reset()
    agent.init_game_setting()
    writer.append_data(env.env.render())
    terminated = truncated = False
    while not terminated and not truncated:
        action = agent.make_action(state, test=True)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        writer.append_data(env.env.render())

writer.close()
env.close()
print(f'Recorded combined video: {video_path}; five-life reward: {total_reward}')
