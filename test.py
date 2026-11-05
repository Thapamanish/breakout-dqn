"""

### NOTICE ###
You DO NOT need to upload this file

"""

import os
import argparse
import numpy as np
from environment import Environment
import time
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

seed = 11037

def test(agent, env, total_episodes=100, record_video=False):
    rewards = []
    env.reset(seed=seed)

    vid = None  # Initialize vid to None to ensure it's accessible outside the if block
    if record_video:
        os.makedirs("test_videos", exist_ok=True)
        vid = RecordVideo(env=env.env, video_folder="test_videos", name_prefix="test_vid", episode_trigger=lambda ep: True)
    start_time = time.time()
    
    
    for _ in tqdm(range(total_episodes)):
        episode_reward = 0.0
        truncated = False
        for _ in range(5):  # Run each episode for 5 lives
            state = env.reset()
            agent.init_game_setting()
            terminated = False

            # playing one game (1 life)
            while not terminated and not truncated:
                action = agent.make_action(state, test=True)
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward


            if truncated:
                break

        rewards.append(episode_reward)

    env.close()

    print('Run %d episodes for 5 lives each' % (total_episodes))
    print('Mean:', np.mean(rewards))
    print('rewards', rewards)
    print('running time', time.time()-start_time)
    if record_video:
        print('Videos saved to ./videos/')
