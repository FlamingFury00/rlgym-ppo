# RLGym-PPO
A vectorized implementation of PPO and Muesli algorithms for use with [RLGym](rlgym.org).

## ALGORITHMS

### PPO (Proximal Policy Optimization)
The traditional and stable reinforcement learning algorithm, perfect for most applications.

### Muesli (Model-Enhanced Policy Optimization) 🚀 NEW!
An advanced algorithm that combines policy optimization with model learning, achieving ~500% sample efficiency improvements over traditional methods. Based on DeepMind's paper "Muesli: Combining Improvements in Policy Optimization".

**Muesli Benefits:**
- **~500% faster learning** compared to PPO
- **Model-based planning** for complex environments like Rocket League
- **Conservative policy updates** for stable training
- **Experience reanalysis** for enhanced sample efficiency
- **Target networks** for training stability

## INSTALLATION
1. install [RLGym-sim](https://github.com/AechPro/rocket-league-gym-sim). 
2. If you would like to use a GPU install [PyTorch with CUDA](https://pytorch.org/get-started/locally/)
3. Install this project via `pip install git+https://github.com/AechPro/rlgym-ppo`

## USAGE

### Basic PPO Usage
Simply import the learner with `from rlgym_ppo import Learner`, pass it a function that will return an RLGym environment
and run the learning algorithm. A simple example follows:
```python
from rlgym_ppo import Learner

def my_rlgym_function():
    import rlgym_sim
    return rlgym_sim.make()

learner = Learner(my_rlgym_env_function)
learner.learn()
```

### Advanced Muesli Usage 🚀
For enhanced performance, use the Muesli algorithm by setting `use_muesli=True`:
```python
from rlgym_ppo import Learner

def my_rlgym_function():
    import rlgym_sim
    return rlgym_sim.make()

learner = Learner(
    my_rlgym_env_function,
    use_muesli=True,
    model_lr=1e-4,
    hidden_state_size=256,
    n_step_unroll=5,
    reanalysis_ratio=0.3
)
learner.learn()
```

### Rocket League Specific Muesli Example
For Rocket League training with optimal Muesli configuration:
```python
from rlgym_ppo import Learner

# See muesli_example.py for complete RocketSim environment setup
learner = Learner(
    build_muesli_rocketsim_env,  # With 8-frame observation stacking
    use_muesli=True,
    
    # Enhanced network sizes for stacked observations  
    policy_layer_sizes=(1024, 512, 256),
    critic_layer_sizes=(1024, 512, 256),
    
    # Muesli-specific parameters optimized for Rocket League
    model_lr=1e-4,
    hidden_state_size=256,
    n_step_unroll=5,
    target_update_rate=0.01,
    conservative_weight=0.1,
    reanalysis_ratio=0.3,
)
learner.learn()
```

Note that users must implement a function to configure Rocket League (or RocketSim) in RLGym that returns an 
RLGym environment. See the `example.py` file for PPO examples and `muesli_example.py` for advanced Muesli examples with RocketSim integration.
