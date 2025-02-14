# Project Architecture

This document provides an overview of the architecture and key components of the `rlgym-ppo` project. The project implements a multi-processed Proximal Policy Optimization (PPO) algorithm tailored for reinforcement learning in Rocket League environments. Below, we detail the main components and their roles in the system.

---

## 1. Proximal Policy Optimization (PPO) Implementation

PPO is a state-of-the-art reinforcement learning algorithm that balances exploration and exploitation by optimizing a clipped surrogate objective function. It is designed to be stable and efficient for large-scale training.

### Key Features:
- **Policy Network**: The policy network outputs actions based on observations. For continuous action spaces, it uses a Gaussian distribution parameterized by mean and variance.
- **Value Network**: The value network estimates the expected return (value) of a given state, aiding in advantage estimation.
- **Clipped Objective**: PPO uses a clipped objective to prevent large updates, ensuring stable training.
- **Entropy Regularization**: Encourages exploration by penalizing overly deterministic policies.
- **Gradient Clipping**: Limits the magnitude of gradients to improve stability.
- **Learning Rate Scheduling**: Dynamically adjusts learning rates for better convergence.

### Implementation Details:
- The PPO implementation is encapsulated in the `PPOLearner` class (`rlgym_ppo/ppo/ppo_learner.py`).
- The policy network is implemented in `ContinuousPolicy`, `MultiDiscreteFF`, and `DiscreteFF` classes, depending on the action space type.
- The value network is implemented in the `ValueEstimator` class.
- The `PPOLearner.learn()` method orchestrates the training process, including data sampling, loss computation, and optimization.

---

## 2. Batched Agent System

The batched agent system enables efficient interaction between multiple agents and the environment using a multi-process architecture. This design allows for parallel data collection, significantly speeding up training.

### Key Features:
- **Multi-Process Architecture**: Each agent runs in its own process, interacting with an environment instance.
- **Shared Memory**: Agents use shared memory for efficient data transfer between processes.
- **Process Synchronization**: The `BatchedAgentManager` class coordinates communication and ensures data consistency.
- **Metrics Collection**: Custom metrics can be collected and logged for analysis.

### Implementation Details:
- The `BatchedAgentManager` class (`rlgym_ppo/batched_agents/batched_agent_manager.py`) manages agent processes, collects timesteps, and organizes trajectories.
- The `batched_agent_process` function (`rlgym_ppo/batched_agents/batched_agent.py`) defines the behavior of individual agent processes.
- Communication between processes is optimized using sockets and shared memory buffers.

---

## 3. Experience Buffer

The experience buffer stores interaction data (states, actions, rewards, etc.) collected by agents. This data is used to train the PPO algorithm.

### Key Features:
- **FIFO Behavior**: Oldest data is discarded when the buffer reaches its maximum size.
- **Shuffling**: Data is shuffled to ensure diverse and unbiased training batches.
- **Efficient Memory Usage**: The buffer uses a deque for fast appends and pops.

### Implementation Details:
- The `ExperienceBuffer` class (`rlgym_ppo/ppo/experience_buffer.py`) handles data storage and sampling.
- The `submit_experience` method adds new data to the buffer.
- The `get_all_batches_shuffled` method yields shuffled batches for training.

---

## 4. Usage Instructions

### Setting Up the Environment
- Define a function to create an RLGym environment. Examples can be found in `example.py` and `rlgym_v2_example.py`.
- Use the `Learner` class to initialize and run the PPO training loop.

### Key Configuration Parameters
- **Hyperparameters**: Configure PPO-specific parameters like `ppo_epochs`, `ppo_batch_size`, and `ppo_clip_range` in the `Learner` class.
- **Process Management**: Adjust the number of processes (`n_proc`) and inference size (`min_inference_size`) for optimal performance.
- **Logging**: Enable logging to Weights & Biases (`log_to_wandb`) for detailed metrics tracking.

### Example Usage
Refer to the `example.py` file for a complete example of setting up and running the training loop:
```python
from rlgym_ppo import Learner

def build_env():
    # Define your RLGym environment here
    pass

learner = Learner(build_env, n_proc=32, ppo_batch_size=50000, ts_per_iteration=50000)
learner.learn()
```

---

## References
- **PPO Algorithm**: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- **Codebase**: [GitHub Repository](https://github.com/AechPro/rlgym-ppo)
- **RLGym**: [RLGym Documentation](https://rlgym.org)

