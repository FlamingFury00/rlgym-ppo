import numpy as np
import torch
import torch.nn as nn


class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device

        assert (
            len(layer_sizes) > 0
        ), "At least one layer must be specified to build the neural network!"

        layers = []
        prev_size = input_shape
        for idx, size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_size, size))

            # Apply Kaiming initialization to the layer
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity="relu")

            if idx < len(layer_sizes) - 1:
                # Batch Normalization
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU(inplace=True))

            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_actions))
        # Apply Kaiming initialization to the final layer
        nn.init.kaiming_normal_(layers[-1].weight, nonlinearity="linear")

        self.model = nn.Sequential(*layers).to(self.device)

        for i, layer in enumerate(self.model):
            if hasattr(layer, "weight"):
                print(f"Layer {i}: {layer} with weight shape {layer.weight.shape}")

        self.n_actions = n_actions

    def get_output(self, obs):
        # Ensure the input is a torch.Tensor
        if not isinstance(obs, torch.Tensor):
            obs = (
                np.array(obs, dtype=np.float32)
                if not isinstance(obs, np.ndarray)
                else obs
            )
            obs = torch.tensor(
                obs, dtype=torch.float32, device=self.model[0].weight.device
            )

        if obs.dim() == 3 and obs.shape[1] == 1:
            obs = obs.squeeze(1)

        # Perform the forward pass
        return self.model(obs)

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            logits = self.get_output(obs)
            probs = nn.functional.softmax(logits, dim=-1)

            if deterministic:
                return probs.argmax(dim=-1).cpu().numpy(), torch.tensor(
                    0, dtype=torch.float32
                )

            distribution = torch.distributions.Categorical(probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

        return action.cpu(), log_prob.cpu()

    def get_backprop_data(self, obs, acts):
        logits = self.get_output(obs)
        probs = nn.functional.softmax(logits, dim=-1)

        log_probs = torch.log(probs)

        # Ensure acts is of type torch.int64 for gather
        action_log_probs = log_probs.gather(-1, acts.long().unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs * probs).sum(dim=-1)

        return action_log_probs.to(self.device), entropy.mean().to(self.device)
