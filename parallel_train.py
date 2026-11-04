#!/usr/bin/env python3
"""Parallel Atari collector for a separate leaderboard experiment.

This leaves the original Agent_DQN and its checkpoint format unchanged. It
uses multiple subprocess environments to keep the learner supplied with
transitions and DataParallel when multiple CUDA devices are visible.
"""
import argparse
import os
import random
from collections import deque

import gymnasium as gym
import ale_py  # Registers the Atari environments used by this project.
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from atari_wrapper import make_wrap_atari
from dqn_model import DQN


class PrioritizedReplayBuffer:
    """A compact proportional prioritized replay buffer.

    Transitions are stored exactly as before, while priorities are kept in a
    separate NumPy array. New transitions receive the current maximum priority
    so that they are sampled at least once before their TD error is known.
    """

    def __init__(self, capacity, alpha=0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.data = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.next_idx = 0

    def __len__(self):
        return len(self.data)

    def append(self, transition):
        maximum = float(self.priorities[:len(self.data)].max()) if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.next_idx] = transition
        self.priorities[self.next_idx] = max(maximum, 1e-6)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size, beta):
        size = len(self.data)
        scaled = self.priorities[:size] ** self.alpha
        probabilities = scaled / scaled.sum()
        indices = np.random.choice(size, size=batch_size, replace=False, p=probabilities)
        weights = (size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        batch = [self.data[i] for i in indices]
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        self.priorities[np.asarray(indices)] = np.maximum(np.asarray(priorities), 1e-6)


class NStepAccumulator:
    """Turn one environment's one-step transitions into n-step samples."""

    def __init__(self, n_step, gamma):
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.transitions = deque()

    def _build_sample(self):
        state, action, _, _, _ = self.transitions[0]
        total_reward = 0.0
        next_state = None
        done = False
        steps = 0
        for _, _, reward, candidate_next_state, candidate_done in self.transitions:
            total_reward += (self.gamma ** steps) * reward
            next_state = candidate_next_state
            steps += 1
            done = candidate_done
            if done or steps == self.n_step:
                break
        return state, action, total_reward, next_state, done, steps

    def append(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        samples = []
        if len(self.transitions) >= self.n_step:
            samples.append(self._build_sample())
            self.transitions.popleft()
        if done:
            while self.transitions:
                samples.append(self._build_sample())
                self.transitions.popleft()
        return samples


def parse_args():
    p = argparse.ArgumentParser()
    # Keep the default below the JupyterHub stability limit. More workers can
    # be requested explicitly after a stable run is confirmed.
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--gpu_count', type=int, default=1,
                   help='Number of visible GPUs to use (default: 1)')
    p.add_argument('--max_steps', type=int, default=5_000_000)
    p.add_argument('--resume', default='',
                   help='Optional checkpoint to continue; omit for a clean run')
    p.add_argument('--output', default='dqn_3step_5m.pth')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--replay_capacity', type=int, default=100_000)
    p.add_argument('--learning_starts', type=int, default=20_000)
    p.add_argument('--train_frequency', type=int, default=4)
    p.add_argument('--target_update_frequency', type=int, default=2_500)
    p.add_argument('--learning_rate', type=float, default=6.25e-5)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--n_step', type=int, default=3,
                   help='Number of discounted rewards per replay target')
    p.add_argument('--epsilon_decay_steps', type=int, default=1_000_000)
    p.add_argument('--prioritized_replay', action='store_true',
                   help='Use proportional prioritized experience replay')
    p.add_argument('--per_alpha', type=float, default=0.6)
    p.add_argument('--per_beta_start', type=float, default=0.4)
    p.add_argument('--per_beta_steps', type=int, default=5_000_000)
    p.add_argument('--checkpoint_interval', type=int, default=250_000)
    return p.parse_args()


def make_env():
    return make_wrap_atari('BreakoutNoFrameskip-v4', clip_rewards=True,
                           render_mode=None)


def state_batch(x, device):
    return torch.as_tensor(x, dtype=torch.uint8, device=device).permute(0, 3, 1, 2)


def save(path, online, target, optimizer, steps, gradients, args):
    net = online.module if isinstance(online, torch.nn.DataParallel) else online
    target_net = target.module if isinstance(target, torch.nn.DataParallel) else target
    torch.save({'online_net': net.state_dict(), 'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(), 'total_steps': steps,
                'gradient_steps': gradients, 'config': vars(args)}, path)


def main():
    args = parse_args()
    if args.n_step < 1:
        raise ValueError('--n_step must be at least 1')
    random.seed(595); np.random.seed(595); torch.manual_seed(595)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visible_gpu_count = torch.cuda.device_count() if device.type == 'cuda' else 0
    gpu_count = min(args.gpu_count, visible_gpu_count)
    print(f'Using device: {device}; visible CUDA devices: {visible_gpu_count}; '
          f'using GPUs: {gpu_count}; workers: {args.workers}; '
          f'n-step: {args.n_step}')

    online = DQN(4, 4).to(device)
    target = DQN(4, 4).to(device)
    # A clean run starts with identical online and target networks.  A resume
    # below then restores its saved target network when one is available.
    target.load_state_dict(online.state_dict())
    steps = gradients = 0
    if args.resume and os.path.exists(args.resume):
        c = torch.load(args.resume, map_location=device)
        online.load_state_dict(c.get('online_net', c))
        target.load_state_dict(c.get('target_net', c.get('online_net', c)))
        steps = int(c.get('total_steps', 0)); gradients = int(c.get('gradient_steps', 0))
        print(f'Resumed {args.resume} at {steps} environment steps')

    if gpu_count > 1:
        ids = list(range(gpu_count))
        online = torch.nn.DataParallel(online, device_ids=ids, output_device=ids[0])
        target = torch.nn.DataParallel(target, device_ids=ids, output_device=ids[0])
        print(f'DataParallel enabled across CUDA devices {ids}')
    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=args.learning_rate, eps=1.5e-4)
    if 'c' in locals() and 'optimizer' in c:
        try: optimizer.load_state_dict(c['optimizer'])
        except ValueError: print('Optimizer state not loaded; using fresh optimizer state')

    memory = (PrioritizedReplayBuffer(args.replay_capacity, args.per_alpha)
              if args.prioritized_replay else deque(maxlen=args.replay_capacity))
    n_step_accumulators = [NStepAccumulator(args.n_step, args.gamma)
                           for _ in range(args.workers)]
    env = gym.vector.AsyncVectorEnv([make_env for _ in range(args.workers)],
                                    shared_memory=False)
    states, _ = env.reset(seed=595)
    last_save = steps
    while steps < args.max_steps:
        fraction = min(1.0, steps / float(args.epsilon_decay_steps))
        epsilon = 1.0 - fraction * 0.99
        with torch.no_grad():
            q = online(state_batch(states, device))
            greedy = q.argmax(1).detach().cpu().numpy()
        actions = np.where(np.random.random(args.workers) < epsilon,
                           np.random.randint(0, 4, args.workers), greedy)
        next_states, rewards, terminated, truncated, infos = env.step(actions)
        for i in range(args.workers):
            done = bool(terminated[i] or truncated[i])
            # Vector environments auto-reset completed slots.  The final
            # observation is the true successor for a terminal replay sample.
            successor = next_states[i]
            if done and infos.get('_final_observation', np.zeros(args.workers, dtype=bool))[i]:
                successor = infos['final_observation'][i]
            samples = n_step_accumulators[i].append(
                states[i].copy(), int(actions[i]), float(rewards[i]),
                np.asarray(successor).copy(), done)
            for sample in samples:
                memory.append(sample)
        states = next_states
        steps += args.workers

        if steps >= args.learning_starts and steps % args.train_frequency < args.workers:
            if len(memory) >= args.batch_size:
                if args.prioritized_replay:
                    beta_fraction = min(1.0, steps / float(args.per_beta_steps))
                    beta = args.per_beta_start + beta_fraction * (1.0 - args.per_beta_start)
                    batch, sample_indices, sample_weights = memory.sample(args.batch_size, beta)
                else:
                    batch = random.sample(memory, args.batch_size)
                s, a, r, ns, d, horizons = zip(*batch)
                s = state_batch(np.stack(s), device)
                ns = state_batch(np.stack(ns), device)
                a = torch.as_tensor(a, dtype=torch.long, device=device).unsqueeze(1)
                r = torch.as_tensor(r, dtype=torch.float32, device=device)
                d = torch.as_tensor(d, dtype=torch.float32, device=device)
                horizons = torch.as_tensor(horizons, dtype=torch.float32, device=device)
                current = online(s).gather(1, a).squeeze(1)
                with torch.no_grad():
                    na = online(ns).argmax(1, keepdim=True)
                    nq = target(ns).gather(1, na).squeeze(1)
                    discounts = torch.pow(torch.full_like(horizons, args.gamma), horizons)
                    expected = r + discounts * nq * (1.0 - d)
                td_errors = current - expected
                if args.prioritized_replay:
                    weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
                    loss = (weights * F.smooth_l1_loss(current, expected, reduction='none')).mean()
                else:
                    loss = F.smooth_l1_loss(current, expected)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step(); gradients += 1
                if args.prioritized_replay:
                    memory.update_priorities(sample_indices, td_errors.detach().abs().cpu().numpy() + 1e-5)
                if gradients % args.target_update_frequency == 0:
                    target.load_state_dict(online.state_dict())

        if steps - last_save >= args.checkpoint_interval:
            save(args.output, online, target, optimizer, steps, gradients, args)
            last_save = steps
            print(f'Steps {steps}/{args.max_steps} | replay {len(memory)} | epsilon {epsilon:.3f}')

    save(args.output, online, target, optimizer, steps, gradients, args)
    env.close()
    print(f'Parallel training complete at {steps} steps; saved {args.output}')


if __name__ == '__main__':
    main()
