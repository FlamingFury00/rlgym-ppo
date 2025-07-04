"""
Reward Model for Muesli Algorithm

This module implements the reward model that learns to predict
the immediate reward given a hidden state and action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """
    Reward model that predicts immediate reward given hidden state and action.

    This is part of Muesli's model learning component that helps improve
    sample efficiency by learning to predict rewards.
    """

    def __init__(
        self,
        hidden_state_size,
        action_size,
        layer_sizes,
        device,
        categorical=False,
        reward_range=(-1, 1),
        num_atoms=51,
    ):
        """
        Initialize the reward model.

        Args:
            hidden_state_size (int): Size of the hidden state representation
            action_size (int): Size of the action space
            layer_sizes (list): List of layer sizes for the network
            device (torch.device): Device to run computations on
            categorical (bool): Whether to use categorical representation
            reward_range (tuple): Range of possible rewards for categorical representation
            num_atoms (int): Number of atoms for categorical distribution
        """
        super(RewardModel, self).__init__()

        self.device = device
        self.hidden_state_size = hidden_state_size
        self.action_size = action_size
        self.categorical = categorical
        self.reward_range = reward_range
        self.num_atoms = num_atoms

        # Declare tensor attributes for type checking
        self.reward_support: torch.Tensor

        # Input is concatenation of hidden state and action
        input_size = hidden_state_size + action_size

        # Build the neural network
        assert len(layer_sizes) > 0, "At least one layer must be specified"

        layers = []
        prev_size = input_size

        for size in layer_sizes:
            layers.extend([nn.Linear(prev_size, size), nn.ReLU()])
            prev_size = size

        # Output layer
        if categorical:
            # Categorical representation: output probability distribution over reward values
            layers.append(nn.Linear(prev_size, num_atoms))
            layers.append(nn.Softmax(dim=-1))
            self._setup_categorical_support()
        else:
            # Scalar representation: output single reward value
            layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _setup_categorical_support(self):
        """Setup support for categorical reward representation."""
        min_reward, max_reward = self.reward_range
        support_tensor = torch.linspace(min_reward, max_reward, self.num_atoms)
        self.register_buffer("reward_support", support_tensor)

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_state, action):
        """
        Predict the immediate reward.

        Args:
            hidden_state (torch.Tensor): Current hidden state [batch_size, hidden_state_size]
            action (torch.Tensor): Action taken [batch_size, action_size] or [batch_size]

        Returns:
            torch.Tensor: Predicted reward(s)
        """
        # Ensure tensors are on the correct device
        hidden_state = hidden_state.to(self.device)
        action = action.to(self.device)

        # Ensure action has correct dimensions - if 1D, expand to 2D
        if len(action.shape) == 1:
            action = action.unsqueeze(-1)

        # Concatenate hidden state and action
        combined_input = torch.cat([hidden_state, action], dim=-1)

        # Predict reward
        reward_output = self.network(combined_input)

        if self.categorical:
            # Return probability distribution over reward values
            return reward_output
        else:
            # Return scalar reward prediction
            return reward_output.squeeze(-1)

    def compute_loss(self, predicted_reward, target_reward):
        """
        Compute the reward model loss.

        Args:
            predicted_reward (torch.Tensor): Predicted rewards
            target_reward (torch.Tensor): Target rewards

        Returns:
            torch.Tensor: Loss between predicted and target rewards
        """
        if self.categorical:
            # Convert target rewards to categorical distribution
            target_distribution = self._scalar_to_categorical(target_reward)
            # Use cross-entropy loss for categorical predictions
            return F.cross_entropy(predicted_reward, target_distribution)
        else:
            # Use MSE loss for scalar predictions
            return F.mse_loss(predicted_reward, target_reward)

    def _scalar_to_categorical(self, rewards):
        """
        Convert scalar rewards to categorical distribution targets.

        Args:
            rewards (torch.Tensor): Scalar reward values

        Returns:
            torch.Tensor: Categorical distribution targets
        """
        rewards = torch.clamp(rewards, self.reward_range[0], self.reward_range[1])

        # Find nearest support points
        distances = torch.abs(rewards.unsqueeze(-1) - self.reward_support.unsqueeze(0))
        indices = torch.argmin(distances, dim=-1)

        # Create one-hot encoding
        batch_size = rewards.size(0)
        categorical_targets = torch.zeros(
            batch_size, self.num_atoms, device=self.device
        )
        categorical_targets.scatter_(1, indices.unsqueeze(1), 1.0)

        return categorical_targets

    def categorical_to_scalar(self, categorical_reward):
        """
        Convert categorical reward prediction to scalar value.

        Args:
            categorical_reward (torch.Tensor): Categorical reward distribution

        Returns:
            torch.Tensor: Scalar reward values
        """
        if not self.categorical:
            return categorical_reward

        # Compute expected value of the categorical distribution
        return torch.sum(categorical_reward * self.reward_support.unsqueeze(0), dim=-1)


class CombinedPredictionHead(nn.Module):
    """
    Combined prediction head that outputs policy, value, and reward predictions.

    This follows the MuZero architecture pattern but adapted for Muesli's requirements.
    """

    def __init__(
        self,
        hidden_state_size,
        action_size,
        layer_sizes,
        device,
        categorical_value=False,
        value_range=(-10, 10),
        num_atoms=51,
    ):
        """
        Initialize the combined prediction head.

        Args:
            hidden_state_size (int): Size of the hidden state representation
            action_size (int): Size of the action space
            layer_sizes (list): List of layer sizes for shared layers
            device (torch.device): Device to run computations on
            categorical_value (bool): Whether to use categorical value representation
            value_range (tuple): Range of possible values for categorical representation
            num_atoms (int): Number of atoms for categorical distribution
        """
        super(CombinedPredictionHead, self).__init__()

        self.device = device
        self.categorical_value = categorical_value
        self.value_range = value_range
        self.num_atoms = num_atoms

        # Declare tensor attributes for type checking
        self.value_support: torch.Tensor

        # Shared layers
        shared_layers = []
        prev_size = hidden_state_size

        for size in layer_sizes:
            shared_layers.extend([nn.Linear(prev_size, size), nn.ReLU()])
            prev_size = size

        self.shared_network = nn.Sequential(*shared_layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(prev_size, action_size), nn.Softmax(dim=-1)
        )

        # Value head
        if categorical_value:
            self.value_head = nn.Sequential(
                nn.Linear(prev_size, num_atoms), nn.Softmax(dim=-1)
            )
            self._setup_value_support()
        else:
            self.value_head = nn.Linear(prev_size, 1)

        # Initialize weights
        self._initialize_weights()

    def _setup_value_support(self):
        """Setup support for categorical value representation."""
        min_value, max_value = self.value_range
        self.register_buffer(
            "value_support", torch.linspace(min_value, max_value, self.num_atoms)
        )

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_state):
        """
        Forward pass through the prediction head.

        Args:
            hidden_state (torch.Tensor): Hidden state [batch_size, hidden_state_size]

        Returns:
            tuple: (policy_logits, value_prediction)
        """
        hidden_state = hidden_state.to(self.device)

        # Shared processing
        shared_features = self.shared_network(hidden_state)

        # Policy prediction
        policy_logits = self.policy_head(shared_features)

        # Value prediction
        value_output = self.value_head(shared_features)

        if self.categorical_value:
            return policy_logits, value_output
        else:
            return policy_logits, value_output.squeeze(-1)

    def categorical_value_to_scalar(self, categorical_value):
        """
        Convert categorical value prediction to scalar value.

        Args:
            categorical_value (torch.Tensor): Categorical value distribution

        Returns:
            torch.Tensor: Scalar value
        """
        if not self.categorical_value:
            return categorical_value

        # Compute expected value of the categorical distribution
        return torch.sum(categorical_value * self.value_support.unsqueeze(0), dim=-1)
