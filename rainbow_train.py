#!/usr/bin/env python3
"""Train a course-compatible Rainbow-lite (C51 + NoisyNet) Breakout agent.

The output checkpoint can be evaluated directly with ``main.py --test_dqn``.
Only the model and agent implementation are needed by the course evaluator;
this script is a faster vectorized training driver for experiments.
"""
import argparse
import os
import random

import gymnasium as gym
import ale_py  # Registers Atari environments.
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from atari_wrapper import make_wrap_atari
from dqn_model import RainbowDQN


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.data = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.next_index = 0

    def __len__(self):
        return len(self.data)

    def append(self, transition):
        maximum = float(self.priorities[:len(self.data)].max()) if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.next_index] = transition
        self.priorities[self.next_index] = max(maximum, 1e-6)
        self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size, beta):
        priorities = self.priorities[:len(self.data)] ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.data), batch_size, replace=False, p=probabilities)
        weights = (len(self.data) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return [self.data[i] for i in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        self.priorities[np.asarray(indices)] = np.maximum(np.asarray(priorities), 1e-6)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=5_000_000)
    parser.add_argument('--output', default='rainbow_c51_noisy_5m.pth')
    parser.add_argument('--resume', default='',
                        help='Checkpoint to resume after an interrupted run')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_capacity', type=int, default=100_000)
    parser.add_argument('--learning_starts', type=int, default=20_000)
    parser.add_argument('--train_frequency', type=int, default=4)
    parser.add_argument('--target_update_frequency', type=int, default=8_000,
                        help='Gradient updates between target-network copies')
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--v_min', type=float, default=-10.0)
    parser.add_argument('--v_max', type=float, default=10.0)
    parser.add_argument('--per_alpha', type=float, default=0.5)
    parser.add_argument('--per_beta_start', type=float, default=0.4)
    parser.add_argument('--per_beta_steps', type=int, default=5_000_000)
    parser.add_argument('--checkpoint_interval', type=int, default=250_000)
    return parser.parse_args()


def make_env():
    return make_wrap_atari('BreakoutNoFrameskip-v4', clip_rewards=True, render_mode=None)


def states_to_tensor(states, device):
    return torch.as_tensor(states, dtype=torch.uint8, device=device).permute(0, 3, 1, 2)


def checkpoint_path(output, steps):
    root, extension = os.path.splitext(output)
    return f'{root}_{steps // 1000}k{extension or ".pth"}'


def save_checkpoint(path, online, target, optimizer, steps, gradients, args):
    torch.save({
        'algorithm': 'rainbow_lite_c51_noisy',
        'online_net': online.state_dict(),
        'target_net': target.state_dict(),
        'optimizer': optimizer.state_dict(),
        'total_steps': int(steps),
        'gradient_steps': int(gradients),
        'config': vars(args),
    }, path)


def project_distribution(next_distribution, rewards, dones, support, gamma, v_min, v_max):
    """Project a Bellman-updated categorical distribution onto fixed C51 atoms."""
    batch_size, num_atoms = next_distribution.shape
    delta_z = (v_max - v_min) / (num_atoms - 1)
    target_atoms = rewards.unsqueeze(1) + gamma * (1.0 - dones.unsqueeze(1)) * support.unsqueeze(0)
    target_atoms = target_atoms.clamp(v_min, v_max)
    positions = (target_atoms - v_min) / delta_z
    lower = positions.floor().long()
    upper = positions.ceil().long()
    projected = torch.zeros_like(next_distribution)
    projected.scatter_add_(1, lower, next_distribution * (upper.float() - positions))
    projected.scatter_add_(1, upper, next_distribution * (positions - lower.float()))
    projected.scatter_add_(1, lower, next_distribution * (lower == upper).float())
    return projected


def main():
    args = parse_args()
    if args.num_atoms < 2 or args.v_min >= args.v_max:
        raise ValueError('Use at least two atoms and require v_min < v_max')
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
    random.seed(595)
    np.random.seed(595)
    torch.manual_seed(595)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Rainbow-lite on {device}; workers={args.workers}; C51 atoms={args.num_atoms}')

    online = RainbowDQN(4, 4, args.num_atoms, args.v_min, args.v_max).to(device)
    target = RainbowDQN(4, 4, args.num_atoms, args.v_min, args.v_max).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=args.learning_rate, eps=1.5e-4)
    steps = gradients = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get('algorithm') != 'rainbow_lite_c51_noisy':
            raise ValueError(f'{args.resume} is not a Rainbow-lite checkpoint')
        online.load_state_dict(checkpoint['online_net'])
        target.load_state_dict(checkpoint['target_net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = int(checkpoint['total_steps'])
        gradients = int(checkpoint['gradient_steps'])
        print(f'Resumed {args.resume} at {steps} environment steps')
    online.train()
    target.train()
    replay = PrioritizedReplayBuffer(args.replay_capacity, args.per_alpha)
    env = gym.vector.AsyncVectorEnv([make_env for _ in range(args.workers)], shared_memory=False)
    states, _ = env.reset(seed=595)
    last_save = steps

    while steps < args.max_steps:
        online.reset_noise()
        with torch.no_grad():
            actions = online(states_to_tensor(states, device)).argmax(1).cpu().numpy()
        next_states, rewards, terminated, truncated, infos = env.step(actions)
        for index in range(args.workers):
            done = bool(terminated[index] or truncated[index])
            successor = next_states[index]
            if done and infos.get('_final_observation', np.zeros(args.workers, dtype=bool))[index]:
                successor = infos['final_observation'][index]
            replay.append((states[index].copy(), int(actions[index]), float(rewards[index]),
                           np.asarray(successor).copy(), done))
        states = next_states
        steps += args.workers

        # Replay is intentionally not serialized. After a resume, rebuild a
        # diverse warm-up buffer before applying further gradient updates.
        if (steps < args.learning_starts or len(replay) < args.learning_starts or
                steps % args.train_frequency >= args.workers):
            continue

        beta_fraction = min(1.0, steps / float(args.per_beta_steps))
        beta = args.per_beta_start + beta_fraction * (1.0 - args.per_beta_start)
        batch, indices, weights = replay.sample(args.batch_size, beta)
        state_batch, actions_batch, rewards_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = states_to_tensor(np.stack(state_batch), device)
        next_state_batch = states_to_tensor(np.stack(next_state_batch), device)
        actions_batch = torch.as_tensor(actions_batch, dtype=torch.long, device=device)
        rewards_batch = torch.as_tensor(rewards_batch, dtype=torch.float32, device=device)
        done_batch = torch.as_tensor(done_batch, dtype=torch.float32, device=device)
        importance_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

        online.reset_noise()
        target.reset_noise()
        distribution = online.dist(state_batch)
        chosen_distribution = distribution[torch.arange(args.batch_size, device=device), actions_batch]
        with torch.no_grad():
            next_actions = online(next_state_batch).argmax(1)
            next_distribution = target.dist(next_state_batch)[
                torch.arange(args.batch_size, device=device), next_actions]
            projected = project_distribution(next_distribution, rewards_batch, done_batch,
                                             online.support, args.gamma, args.v_min, args.v_max)
        per_sample_loss = -(projected * chosen_distribution.clamp_min(1e-6).log()).sum(1)
        loss = (importance_weights * per_sample_loss).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        optimizer.step()
        replay.update_priorities(indices, per_sample_loss.detach().cpu().numpy() + 1e-6)
        gradients += 1
        if gradients % args.target_update_frequency == 0:
            target.load_state_dict(online.state_dict())

        if steps - last_save >= args.checkpoint_interval:
            snapshot = checkpoint_path(args.output, steps)
            save_checkpoint(snapshot, online, target, optimizer, steps, gradients, args)
            save_checkpoint(args.output, online, target, optimizer, steps, gradients, args)
            last_save = steps
            print(f'Steps {steps}/{args.max_steps} | replay {len(replay)} | '
                  f'loss {loss.item():.4f} | saved {snapshot}')

    save_checkpoint(args.output, online, target, optimizer, steps, gradients, args)
    env.close()
    print(f'Rainbow-lite complete at {steps} steps; saved {args.output}')


if __name__ == '__main__':
    main()
