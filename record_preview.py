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
parser.add_argument('--episode_index', type=int, default=0,
                    help='1-based episode to record; 0 automatically selects the best')
parser = add_arguments(parser)
args = parser.parse_args()
args.test_dqn = True
args.train_dqn_again = False

os.makedirs(args.video_folder, exist_ok=True)
env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True,
                  test=True, render_mode='rgb_array')
agent = Agent_DQN(env, args)
env.reset(seed=EVAL_SEED)

video_path = os.path.join(args.video_folder, 'breakout_combined.mp4')

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

# With episode_index=0, evaluate first and choose the actual best episode.
if args.episode_index <= 0:
    scores = []
    env.reset(seed=EVAL_SEED)
    for _ in range(args.eval_episodes):
        score, _ = play_one_game(record=False)
        scores.append(score)
    best_offset = max(range(len(scores)), key=scores.__getitem__)
    args.episode_index = best_offset + 1
    print(f'Best episode selected dynamically: {args.episode_index} '
          f'with score {scores[best_offset]}')
    env.reset(seed=EVAL_SEED)
else:
    env.reset(seed=EVAL_SEED)

# Replay earlier five-life games without recording so the requested game is
# aligned with the same deterministic evaluation sequence.
for _ in range(max(0, args.episode_index - 1)):
    play_one_game(record=False)

total_reward, frames = play_one_game(record=True)
writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)
for frame in frames:
    writer.append_data(frame)

writer.close()
env.close()
print(f'Recorded episode {args.episode_index}: {video_path}; five-life reward: {total_reward}')
