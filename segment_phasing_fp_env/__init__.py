from .segment_phasing_fp_env import SegmentPhasingFPEnv

from gymnasium.envs.registration import register

__all__ = [SegmentPhasingFPEnv]

register(id="SegmentPhasingFP-v0",
         entry_point="segment_phasing_fp_env:SegmentPhasingFPEnv")
