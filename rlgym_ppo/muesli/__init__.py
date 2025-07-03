"""
Muesli: Combining Improvements in Policy Optimization

This module implements the Muesli algorithm as described in the DeepMind paper:
"Muesli: Combining Improvements in Policy Optimization" (Hessel et al., 2021)

The algorithm combines regularized policy optimization with model learning as an auxiliary loss,
achieving state-of-the-art performance without requiring deep search like MuZero.
"""

from .muesli_learner import MuesliLearner
from .dynamics_model import DynamicsModel
from .reward_model import RewardModel
from .target_networks import TargetNetworkManager
from .muesli_policy import MuesliPolicy
from .experience_reanalyzer import ExperienceReanalyzer
from .retrace import RetraceEstimator, TrajectoryProcessor
from .muesli_experience_buffer import MuesliExperienceBuffer

__all__ = [
    "MuesliLearner",
    "DynamicsModel",
    "RewardModel",
    "TargetNetworkManager",
    "MuesliPolicy",
    "ExperienceReanalyzer",
    "RetraceEstimator",
    "TrajectoryProcessor",
    "MuesliExperienceBuffer",
]
