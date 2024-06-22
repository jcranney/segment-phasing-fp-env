import gymnasium as gym
import numpy as np
import pygame

import segment_phasing_fp_env  # noqa

from gymnasium.utils.play import play

env = gym.make("SegmentPhasingFP-v0", render_mode="rgb_array")
play(
     env,
     keys_to_action={
        (pygame.K_SPACE,): np.r_[1, 0, 0, 0, 0, 0].astype(np.float32)
     },
     noop=np.zeros(6, dtype=np.float32),
     fps=24
)
