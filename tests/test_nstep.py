"""Dependency-free correctness tests for Rainbow n-step replay construction."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rainbow_utils import NStepAccumulator


def assert_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9), (actual, expected)


def test_full_window():
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    assert accumulator.append('s0', 0, 1, 's1', False) == []
    assert accumulator.append('s1', 1, 2, 's2', False) == []
    sample = accumulator.append('s2', 2, 4, 's3', False)[0]
    state, action, reward, next_state, done, horizon = sample
    assert (state, action, next_state, done, horizon) == ('s0', 0, 's3', False, 3)
    assert_close(reward, 1 + 0.5 * 2 + 0.25 * 4)


def test_terminal_flushes_partial_windows():
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    assert accumulator.append('s0', 0, 1, 's1', False) == []
    samples = accumulator.append('s1', 1, 2, 'terminal', True)
    assert len(samples) == 2
    first, second = samples
    assert first[0:2] == ('s0', 0)
    assert first[3:] == ('terminal', True, 2)
    assert_close(first[2], 2.0)
    assert second[0:2] == ('s1', 1)
    assert second[3:] == ('terminal', True, 1)
    assert_close(second[2], 2.0)


def test_workers_do_not_mix_trajectories():
    left = NStepAccumulator(n_step=2, gamma=0.9)
    right = NStepAccumulator(n_step=2, gamma=0.9)
    assert left.append('left0', 0, 1, 'left1', False) == []
    assert right.append('right0', 1, 5, 'right1', False) == []
    left_sample = left.append('left1', 0, 2, 'left2', False)[0]
    right_sample = right.append('right1', 1, 7, 'right2', False)[0]
    assert left_sample[0] == 'left0' and left_sample[3] == 'left2'
    assert right_sample[0] == 'right0' and right_sample[3] == 'right2'
    assert_close(left_sample[2], 2.8)
    assert_close(right_sample[2], 11.3)


if __name__ == '__main__':
    test_full_window()
    test_terminal_flushes_partial_windows()
    test_workers_do_not_mix_trajectories()
    print('n-step accumulator tests passed')
