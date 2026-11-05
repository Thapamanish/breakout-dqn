"""Small, dependency-free helpers for Rainbow experiments."""
from collections import deque


class NStepAccumulator:
    """Build n-step replay samples for exactly one environment worker.

    Each emitted item is ``(state, action, discounted_reward, next_state,
    done, horizon)``. A terminal transition flushes all partial windows and
    never bootstraps past the terminal state.
    """

    def __init__(self, n_step, gamma):
        if int(n_step) < 1:
            raise ValueError('n_step must be at least one')
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.transitions = deque()

    def _sample_from_front(self):
        state, action, _, _, _ = self.transitions[0]
        reward_sum = 0.0
        next_state = None
        done = False
        horizon = 0
        for _, _, reward, candidate_next_state, candidate_done in self.transitions:
            reward_sum += (self.gamma ** horizon) * reward
            next_state = candidate_next_state
            horizon += 1
            done = bool(candidate_done)
            if done or horizon == self.n_step:
                break
        return state, action, reward_sum, next_state, done, horizon

    def append(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, float(reward), next_state, bool(done)))
        emitted = []
        if len(self.transitions) >= self.n_step:
            emitted.append(self._sample_from_front())
            self.transitions.popleft()
        if done:
            while self.transitions:
                emitted.append(self._sample_from_front())
                self.transitions.popleft()
        return emitted
