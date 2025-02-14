# RLGym-PPO
A vectorized implementation of PPO for use with [RLGym](rlgym.org).

## INSTALLATION
1. Install [RLGym-sim](https://github.com/AechPro/rocket-league-gym-sim) by following the instructions in its repository.
2. Install PyTorch with CUDA for GPU support. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to select the appropriate version for your system.
3. Install this project via `pip install git+https://github.com/AechPro/rlgym-ppo`

## USAGE
To use the library, import the `Learner` class, define a function to create an RLGym environment, and pass it to the `Learner`. Below is a detailed example:
```
from rlgym_ppo import Learner

# Define a function to create a custom RLGym environment
def build_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils.action_parsers import ContinuousAction

    return rlgym_sim.make(
        tick_skip=8,
        team_size=1,
        spawn_opponents=True,
        terminal_conditions=[NoTouchTimeoutCondition(1200), GoalScoredCondition()],
        reward_fn=CombinedReward(
            reward_functions=[
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(team_goal=1, concede=-1, demo=0.1)
            ],
            reward_weights=[0.01, 0.1, 10.0]
        ),
        obs_builder=DefaultObs(),
        action_parser=ContinuousAction()
    )

# Initialize the learner
learner = Learner(
    env_create_function=build_env,
    n_proc=32,
    ppo_batch_size=50000,
    ts_per_iteration=50000,
    exp_buffer_size=150000,
    ppo_minibatch_size=50000,
    ppo_ent_coef=0.001,
    ppo_epochs=1,
    standardize_returns=True,
    standardize_obs=False,
    save_every_ts=100_000,
    timestep_limit=1_000_000_000,
    log_to_wandb=True
)

# Start training
learner.learn()
```

For more examples, refer to the [`example.py`](example.py) file.

Note:
- Users must implement a function to configure Rocket League (or RocketSim) in RLGym that returns an RLGym environment. See the [`example.py`](example.py) file for an example.

---

## Performance Tips
- **Optimize Hyperparameters**: Experiment with `ppo_batch_size`, `ppo_epochs`, and learning rates for better performance.
- **Use GPU**: Ensure PyTorch is installed with CUDA support and set `device="auto"` in the `Learner` class to utilize your GPU.
- **Debugging**: Use fewer processes (`n_proc`) and smaller environments during debugging to reduce complexity.
- **Monitor Training**: Enable logging to Weights & Biases (`log_to_wandb=True`) for detailed metrics tracking.---

## Common Issues and Troubleshooting
1. **Installation Errors**:
   - Ensure all dependencies are installed. Use `pip install -r requirements.txt` to install the required packages.
   - For GPU support, verify that your CUDA version is compatible with the installed PyTorch version.

2. **Runtime Errors**:
   - If the environment fails to initialize, check that RLGym-sim is correctly installed and configured.
   - Ensure the `env_create_function` returns a valid RLGym environment.

3. **Performance Issues**:
   - If training is slow, reduce the number of processes (`n_proc`) or batch size (`ppo_batch_size`).
   - Monitor GPU usage to ensure it is being utilized effectively.

For additional help, refer to the [documentation](docs/architecture.md) or open an issue on the [GitHub repository](https://github.com/AechPro/rlgym-ppo).---

## Documentation
- [Project Architecture](docs/architecture.md): Learn about the system's design and key components.
- [Tutorials and Examples](docs/tutorials.md): Step-by-step guides to get started with `rlgym-ppo`.
