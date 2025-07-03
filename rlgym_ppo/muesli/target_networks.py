"""
Target Networks Manager for Muesli Algorithm

This module implements the target network system used in Muesli for stable training.
Target networks are updated using exponential moving averages to provide stable targets.
"""

import torch
import torch.nn as nn
from copy import deepcopy


class TargetNetworkManager:
    """
    Manages target networks for stable training in the Muesli algorithm.

    Target networks are copies of the main networks that are updated slowly
    using exponential moving averages to provide stable learning targets.
    """

    def __init__(self, tau=0.005, device=None):
        """
        Initialize the target network manager.

        Args:
            tau (float): Target network update rate (exponential moving average coefficient)
            device (torch.device): Device to run computations on
        """
        self.tau = tau
        self.device = device
        self.target_networks = {}
        self.main_networks = {}

    def register_network(self, name, main_network):
        """
        Register a main network and create its corresponding target network.

        Args:
            name (str): Name identifier for the network
            main_network (nn.Module): The main network to create a target for
        """
        # Create a deep copy of the main network as the target
        target_network = deepcopy(main_network)

        # Move to device if specified
        if self.device is not None:
            target_network = target_network.to(self.device)

        # Set target network to evaluation mode
        target_network.eval()

        # Disable gradients for target network
        for param in target_network.parameters():
            param.requires_grad = False

        # Store references
        self.main_networks[name] = main_network
        self.target_networks[name] = target_network

        print(f"Registered target network '{name}' with τ={self.tau}")

    def get_target_network(self, name):
        """
        Get the target network by name.

        Args:
            name (str): Name of the target network

        Returns:
            nn.Module: The target network
        """
        if name not in self.target_networks:
            raise ValueError(
                f"Target network '{name}' not found. Available: {list(self.target_networks.keys())}"
            )
        return self.target_networks[name]

    def update_target_network(self, name):
        """
        Update a specific target network using exponential moving average.

        Args:
            name (str): Name of the network to update
        """
        if name not in self.target_networks:
            raise ValueError(f"Target network '{name}' not found")

        main_net = self.main_networks[name]
        target_net = self.target_networks[name]

        # Exponential moving average update: θ_target = τ * θ_main + (1 - τ) * θ_target
        with torch.no_grad():
            for main_param, target_param in zip(
                main_net.parameters(), target_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * main_param.data + (1.0 - self.tau) * target_param.data
                )

    def update_all_target_networks(self):
        """Update all registered target networks."""
        for name in self.target_networks.keys():
            self.update_target_network(name)

    def hard_update_target_network(self, name):
        """
        Perform a hard update (complete copy) of a target network.

        Args:
            name (str): Name of the network to update
        """
        if name not in self.target_networks:
            raise ValueError(f"Target network '{name}' not found")

        main_net = self.main_networks[name]
        target_net = self.target_networks[name]

        # Hard copy all parameters
        with torch.no_grad():
            for main_param, target_param in zip(
                main_net.parameters(), target_net.parameters()
            ):
                target_param.data.copy_(main_param.data)

    def hard_update_all_target_networks(self):
        """Perform hard updates for all registered target networks."""
        for name in self.target_networks.keys():
            self.hard_update_target_network(name)

    def set_tau(self, new_tau):
        """
        Set a new tau value for target network updates.

        Args:
            new_tau (float): New tau value
        """
        self.tau = new_tau
        print(f"Updated target network τ to {new_tau}")

    def get_target_network_info(self):
        """
        Get information about registered target networks.

        Returns:
            dict: Information about each target network
        """
        info = {}
        for name, target_net in self.target_networks.items():
            param_count = sum(p.numel() for p in target_net.parameters())
            info[name] = {
                "parameter_count": param_count,
                "device": next(target_net.parameters()).device,
                "requires_grad": any(p.requires_grad for p in target_net.parameters()),
            }
        return info


class TargetValueEstimator(nn.Module):
    """
    Target value estimator that uses target networks for computing stable value targets.

    This is used in Muesli for computing multi-step value targets using the learned model.
    """

    def __init__(self, target_network_manager, gamma=0.99):
        """
        Initialize the target value estimator.

        Args:
            target_network_manager (TargetNetworkManager): Manager for target networks
            gamma (float): Discount factor for multi-step returns
        """
        super(TargetValueEstimator, self).__init__()
        self.target_manager = target_network_manager
        self.gamma = gamma

    def compute_n_step_targets(self, states, actions, rewards, dones, n_steps=5):
        """
        Compute n-step value targets using target networks and learned models.

        Args:
            states (torch.Tensor): Initial states
            actions (torch.Tensor): Actions taken [batch_size, n_steps, action_size]
            rewards (torch.Tensor): Immediate rewards [batch_size, n_steps]
            dones (torch.Tensor): Done flags [batch_size, n_steps]
            n_steps (int): Number of steps for n-step returns

        Returns:
            torch.Tensor: N-step value targets
        """
        batch_size = states.size(0)
        device = states.device

        # Initialize return computation
        returns = torch.zeros(batch_size, device=device)
        discount = 1.0

        # Get target networks
        try:
            target_dynamics = self.target_manager.get_target_network("dynamics")
            target_value = self.target_manager.get_target_network("value")
        except ValueError as e:
            raise RuntimeError(f"Required target networks not found: {e}")

        # Simulate n-step trajectory using target networks
        current_state = states
        step_done = torch.zeros(batch_size, device=device, dtype=torch.bool)

        for step in range(n_steps):
            # Check if episode is done
            if step < dones.size(1):
                step_done = dones[:, step]

            # Add immediate reward for this step
            if step < rewards.size(1):
                step_reward = rewards[:, step]
                returns += discount * step_reward * (~step_done).float()

            # If not done, continue simulation
            if step < actions.size(1) and step < n_steps - 1:
                step_action = actions[:, step]

                # Predict next state using target dynamics model
                with torch.no_grad():
                    current_state = target_dynamics(current_state, step_action)

                # Update discount factor
                discount *= self.gamma

                # Stop if all episodes are done
                if step_done.all():
                    break

        # Add final state value estimate using target value network
        if not step_done.all():
            with torch.no_grad():
                final_values = target_value(current_state)
                if hasattr(target_value, "categorical_value_to_scalar"):
                    final_values = target_value.categorical_value_to_scalar(
                        final_values
                    )
                returns += discount * final_values * (~step_done).float()

        return returns

    def compute_reanalyzed_targets(self, stored_experiences, n_steps=5):
        """
        Recompute value targets for stored experiences using updated target networks.

        Args:
            stored_experiences (list): List of stored experience tuples
            n_steps (int): Number of steps for reanalysis

        Returns:
            torch.Tensor: Reanalyzed value targets
        """
        # Extract components from stored experiences
        states = torch.stack([exp[0] for exp in stored_experiences])
        actions = torch.stack([exp[1] for exp in stored_experiences])
        rewards = torch.stack([exp[2] for exp in stored_experiences])
        dones = torch.stack([exp[3] for exp in stored_experiences])

        # Compute new targets using current target networks
        return self.compute_n_step_targets(states, actions, rewards, dones, n_steps)
