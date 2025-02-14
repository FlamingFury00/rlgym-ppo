# RLGym-PPO
A vectorized implementation of PPO for use with [RLGym](rlgym.org).

## INSTALLATION
1. install [RLGym-sim](https://github.com/AechPro/rocket-league-gym-sim). 
2. If you would like to use a GPU install [PyTorch with CUDA](https://pytorch.org/get-started/locally/)
3. Install this project via `pip install git+https://github.com/AechPro/rlgym-ppo`

## ALGORITHM OVERVIEW
RLGym-PPO implements the Proximal Policy Optimization (PPO) algorithm, a popular reinforcement learning method. Key parameters include:
- **ppo_epochs**: Number of epochs to train the policy per iteration.
- **ppo_batch_size**: Number of timesteps per training batch.
- **ppo_minibatch_size**: Size of minibatches for gradient updates.
- **ppo_clip_range**: Clipping range for policy updates to ensure stability.
- **ppo_ent_coef**: Entropy coefficient to encourage exploration.
- **gae_lambda**: Lambda for Generalized Advantage Estimation (GAE).
- **gae_gamma**: Discount factor for future rewards.

## PERFORMANCE OPTIMIZATION
To optimize performance:
1. Use a GPU for faster tensor operations.
2. Increase `n_proc` to utilize multiple CPU cores.
3. Adjust `min_inference_size` to balance inference and collection overhead.
4. Pre-allocate tensors to reduce memory allocation overhead.
5. Use larger batch sizes for more stable updates.

## LEARNING ALGORITHM DETAILS
The learning process involves:
1. Collecting experience from multiple environments in parallel.
2. Calculating advantages and value targets using GAE.
3. Updating the policy and value networks using PPO loss.
4. Logging metrics such as reward, entropy, and KL divergence.

## HARDWARE REQUIREMENTS
Recommended hardware:
- **CPU**: Multi-core processor (e.g., 8+ cores).
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 3060 or better).
- **RAM**: At least 16 GB for large-scale training.

## CONFIGURATION PATTERNS
Example configurations:
- **Small-scale training**:
  ```python
  Learner(env_fn, n_proc=4, ppo_batch_size=10000, ppo_epochs=3)
  ```
- **Large-scale training**:
  ```python
  Learner(env_fn, n_proc=32, ppo_batch_size=50000, ppo_epochs=10)
  ```

## TROUBLESHOOTING
Common issues and solutions:
1. **CUDA out of memory**: Reduce `ppo_batch_size` or `ppo_minibatch_size`.
2. **Slow training**: Ensure GPU is being utilized and increase `n_proc`.
3. **Diverging policy**: Lower `ppo_clip_range` or `ppo_ent_coef`.
4. **Nan values in loss**: Check for invalid rewards or observations.

## USAGE
Simply import the learner with `from rlgym_ppo import Learner`, pass it a function that will return an RLGym environment
and run the learning algorithm. A simple example follows:
```
from rlgym_ppo import Learner

def my_rlgym_function():
    import rlgym_sim
    return rlgym_sim.make()

learner = Learner(my_rlgym_env_function)
learner.learn()
```
Note that users must implement a function to configure Rocket League (or RocketSim) in RLGym that returns an 
RLGym environment. See the `example.py` file for an example of writing such a function.