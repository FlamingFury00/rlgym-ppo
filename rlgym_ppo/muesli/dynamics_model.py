"""
Dynamics Model for Muesli Algorithm

This module implements the dynamics/transition model that learns to predict
the next hidden state given the current hidden state and action.
Based on MuZero-style architecture but simplified for Muesli's one-step lookahead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """
    Dynamics model that predicts next hidden state given current state and action.

    This is a key component of Muesli that learns environment dynamics without
    requiring the full MCTS search of MuZero.
    """

    def __init__(self, hidden_state_size, action_size, layer_sizes, device):
        """
        Initialize the dynamics model.

        Args:
            hidden_state_size (int): Size of the hidden state representation
            action_size (int): Size of the action space
            layer_sizes (list): List of layer sizes for the network
            device (torch.device): Device to run computations on
        """
        super(DynamicsModel, self).__init__()

        self.device = device
        self.hidden_state_size = hidden_state_size
        self.action_size = action_size

        # Input is concatenation of hidden state and action
        input_size = hidden_state_size + action_size

        # Build the neural network
        assert len(layer_sizes) > 0, "At least one layer must be specified"

        layers = []
        prev_size = input_size

        for size in layer_sizes:
            layers.extend([nn.Linear(prev_size, size), nn.ReLU()])
            prev_size = size

        # Output layer produces next hidden state
        layers.append(nn.Linear(prev_size, hidden_state_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_state, action):
        """
        Predict the next hidden state.

        Args:
            hidden_state (torch.Tensor): Current hidden state [batch_size, hidden_state_size]
            action (torch.Tensor): Action taken [batch_size, action_size] or [batch_size]

        Returns:
            torch.Tensor: Predicted next hidden state [batch_size, hidden_state_size]
        """
        # Ensure tensors are on the correct device
        hidden_state = hidden_state.to(self.device)
        action = action.to(self.device)

        # Ensure action has correct dimensions - if 1D, expand to 2D
        if len(action.shape) == 1:
            action = action.unsqueeze(-1)

        # Concatenate hidden state and action
        combined_input = torch.cat([hidden_state, action], dim=-1)

        # Predict next hidden state
        next_hidden_state = self.network(combined_input)

        # Apply min-max scaling to keep hidden states in [-1, 1] range as per Muesli paper
        # This is critical for training stability
        min_vals = next_hidden_state.min(dim=-1, keepdim=True)[0]
        max_vals = next_hidden_state.max(dim=-1, keepdim=True)[0]

        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = torch.where(
            range_vals > 1e-8, range_vals, torch.ones_like(range_vals)
        )

        # Scale to [-1, 1] range: 2 * (x - min) / (max - min) - 1
        next_hidden_state = 2 * (next_hidden_state - min_vals) / range_vals - 1

        return next_hidden_state

    def compute_loss(self, predicted_next_state, target_next_state):
        """
        Compute the dynamics model loss.

        Args:
            predicted_next_state (torch.Tensor): Predicted next hidden states
            target_next_state (torch.Tensor): Target next hidden states

        Returns:
            torch.Tensor: MSE loss between predicted and target states
        """
        return F.mse_loss(predicted_next_state, target_next_state)

    def multi_step_prediction(self, initial_state, actions):
        """
        Perform multi-step prediction using the dynamics model.

        Args:
            initial_state (torch.Tensor): Initial hidden state
            actions (torch.Tensor): Sequence of actions [batch_size, num_steps, action_size]

        Returns:
            list: List of predicted hidden states for each step
        """
        predictions = []
        current_state = initial_state

        for step in range(actions.size(1)):
            action = actions[:, step]
            next_state = self.forward(current_state, action)
            predictions.append(next_state)
            current_state = next_state

        return predictions


class RepresentationModel(nn.Module):
    """
    Representation model that encodes observations into hidden states.

    This converts raw observations into the hidden state representation
    used by the dynamics and prediction models.
    """

    def __init__(self, obs_size, hidden_state_size, layer_sizes, device):
        """
        Initialize the representation model.

        Args:
            obs_size (int): Size of the observation space
            hidden_state_size (int): Size of the hidden state representation
            layer_sizes (list): List of layer sizes for the network
            device (torch.device): Device to run computations on
        """
        super(RepresentationModel, self).__init__()

        self.device = device
        self.obs_size = obs_size
        self.hidden_state_size = hidden_state_size

        # Build the neural network
        assert len(layer_sizes) > 0, "At least one layer must be specified"

        layers = []
        prev_size = obs_size

        for size in layer_sizes:
            layers.extend([nn.Linear(prev_size, size), nn.ReLU()])
            prev_size = size

        # Output layer produces hidden state
        layers.append(nn.Linear(prev_size, hidden_state_size))
        layers.append(nn.Tanh())  # Scale to [-1, 1] range

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observation):
        """
        Encode observation into hidden state.

        Args:
            observation (torch.Tensor): Raw observation [batch_size, obs_size]

        Returns:
            torch.Tensor: Hidden state representation [batch_size, hidden_state_size]
        """
        observation = observation.to(self.device)
        hidden_state = self.network(observation)
        return hidden_state
