from .learner import Learner

# Make Muesli components accessible at package level for advanced users
try:
    from .muesli import (
        MuesliLearner,
        MuesliPolicy,
        DynamicsModel,
        RewardModel,
        TargetNetworkManager,
        RetraceEstimator,
    )

    __all__ = [
        "Learner",
        "MuesliLearner",
        "MuesliPolicy",
        "DynamicsModel",
        "RewardModel",
        "TargetNetworkManager",
        "RetraceEstimator",
    ]
except ImportError:
    # Fallback if Muesli components are not available
    __all__ = ["Learner"]
