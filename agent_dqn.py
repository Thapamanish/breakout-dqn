#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import os
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN, RainbowDQN

SEED = 595

"""Double-DQN agent for the CS654 Breakout project."""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
        self.num_actions = int(env.get_action_space().n)
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.learning_starts = args.learning_starts
        self.train_frequency = args.train_frequency
        self.target_update_frequency = args.target_update_frequency
        self.epsilon_start, self.epsilon_end = args.epsilon_start, args.epsilon_end
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.model_path = args.model_path
        self.is_rainbow = False
        self._build_networks(rainbow=False)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=1.5e-4)
        self.memory = deque(maxlen=args.replay_capacity)
        self.total_steps = self.gradient_steps = 0
        self.losses = []

        # Evaluation-only fallback for the sparse endgame.  The learned policy
        # remains in control normally; this is activated only after a long
        # interval without a positive reward.
        self.escape_stagnation_steps = int(getattr(args, 'escape_stagnation_steps', 0))
        self.escape_cooldown_steps = int(getattr(args, 'escape_cooldown_steps', 600))
        self._escape_enabled = (bool(getattr(args, 'test_dqn', False)) and
                                self.escape_stagnation_steps > 0)
        self._reset_escape_state()

        if args.test_dqn or args.train_dqn_again:
            self.load_checkpoint(required=args.test_dqn)

    def _build_networks(self, rainbow):
        """Create either the legacy DQN or the C51/NoisyNet evaluation model."""
        network_type = RainbowDQN if rainbow else DQN
        self.online_net = network_type(4, self.num_actions).to(self.device)
        self.target_net = network_type(4, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.is_rainbow = bool(rainbow)
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        self._reset_escape_state()

    def _reset_escape_state(self):
        self._steps_since_progress = 0
        self._escape_cooldown = 0
        self._escape_actions = ()
        self._escape_index = 0

    def _start_escape_maneuver(self):
        """Schedule a short paddle sweep to break a repeated endgame path."""
        # BreakoutNoFrameskip-v4 uses 2=RIGHT and 3=LEFT.  The no-op pauses
        # give the ball time to reach the paddle between steering bursts.
        self._escape_actions = (
            (2,) * 18 + (0,) * 4 +
            (3,) * 30 + (0,) * 4 +
            (2,) * 26
        )
        self._escape_index = 0
        self._escape_cooldown = self.escape_cooldown_steps
        self._steps_since_progress = 0

    def observe_transition(self, state, action, reward, next_state,
                           terminated=False, truncated=False):
        """Update endgame detection after an evaluation environment step.

        This hook is optional and is called by ``test.py``.  Training is not
        changed because the fallback is enabled only for test-time agents.
        """
        if not self._escape_enabled:
            return

        if float(reward) > 0.0:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        if self._escape_cooldown > 0:
            self._escape_cooldown -= 1

        if (not terminated and not truncated and
                not self._escape_actions and
                self._escape_cooldown == 0 and
                self._steps_since_progress >= self.escape_stagnation_steps):
            self._start_escape_maneuver()
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        state = np.asarray(observation, dtype=np.uint8)
        if test and self._escape_actions:
            action = int(self._escape_actions[self._escape_index])
            self._escape_index += 1
            if self._escape_index >= len(self._escape_actions):
                self._escape_actions = ()
                self._escape_index = 0
            return action
        if not test and random.random() < self._epsilon():
            return int(self.env.get_random_action())
        with torch.no_grad():
            tensor = torch.as_tensor(state, device=self.device).permute(2, 0, 1).unsqueeze(0)
            return int(self.online_net(tensor).argmax(1).item())
    
    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        self.memory.append((np.asarray(state, dtype=np.uint8).copy(), int(action), float(reward),
                            np.asarray(next_state, dtype=np.uint8).copy(), bool(done)))
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(np.stack(states), device=self.device).permute(0, 3, 1, 2)
        next_states = torch.as_tensor(np.stack(next_states), device=self.device).permute(0, 3, 1, 2)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones
        

    def learn(self):
        """
        Implement your training algorithm here
        """
        if len(self.memory) < max(self.batch_size, self.learning_starts):
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer()
        current_q = self.online_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.losses.append(float(loss.item()))
        return float(loss.item())

    def _epsilon(self):
        fraction = min(1.0, self.total_steps / float(self.epsilon_decay_steps))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def save_checkpoint(self, path=None):
        path = path or self.model_path
        torch.save({'online_net': self.online_net.state_dict(), 'target_net': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'total_steps': self.total_steps,
                    'gradient_steps': self.gradient_steps, 'config': vars(self.args)}, path)

    def load_checkpoint(self, required=False):
        path = next((p for p in (self.model_path, 'dqn_model.pth', 'model.pth') if p and os.path.exists(p)), None)
        if path is None:
            if required:
                raise FileNotFoundError('No trained model found. Train first or pass --model_path.')
            return
        checkpoint = torch.load(path, map_location=self.device)
        is_rainbow = (isinstance(checkpoint, dict) and
                      checkpoint.get('algorithm') == 'rainbow_lite_c51_noisy')
        if is_rainbow and not self.is_rainbow:
            self._build_networks(rainbow=True)
            self.optimizer = optim.Adam(self.online_net.parameters(),
                                        lr=self.args.learning_rate, eps=1.5e-4)
        online = checkpoint.get('online_net', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.online_net.load_state_dict(online)
        self.target_net.load_state_dict(checkpoint.get('target_net', online) if isinstance(checkpoint, dict) else online)
        if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = int(checkpoint.get('total_steps', 0)) if isinstance(checkpoint, dict) else 0
        self.gradient_steps = int(checkpoint.get('gradient_steps', 0)) if isinstance(checkpoint, dict) else 0
        print(f'Loaded trained model from {path} on {self.device}')

    def train(self):
        rewards = deque(maxlen=100)
        state, episode_reward, episode = self.env.reset(seed=SEED), 0.0, 0
        best_mean = -float('inf')
        while episode < self.args.episodes and self.total_steps < self.args.max_steps:
            action = self.make_action(state, test=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            self.push(state, action, reward, next_state, done)
            self.total_steps += 1
            episode_reward += reward
            if self.total_steps >= self.learning_starts and self.total_steps % self.train_frequency == 0:
                self.learn()
            state = self.env.reset() if done else next_state
            if done:
                episode += 1
                rewards.append(episode_reward)
                mean = float(np.mean(rewards))
                # Do not call the first few exploratory episodes the "best"
                # checkpoint; wait until the rolling window is meaningful.
                if len(rewards) >= 100 and mean > best_mean:
                    best_mean = mean
                    self.save_checkpoint(self.args.best_model_path)
                if episode == 1 or episode % self.args.log_interval == 0:
                    print(f'Episode {episode:5d}/{self.args.episodes} | steps {self.total_steps:8d} | '
                          f'100-episode mean {mean:7.2f} | eps {self._epsilon():.3f}')
                if episode % self.args.checkpoint_interval == 0:
                    self.save_checkpoint(self.model_path)
                episode_reward = 0.0
        self.save_checkpoint(self.model_path)
        self.env.close()
        print(f'Training complete: {episode} episodes, {self.total_steps} steps, best mean {best_mean:.2f}')
