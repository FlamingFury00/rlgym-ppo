import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(
            torch.tensor(k.size(-1), dtype=torch.float32, device=x.device)
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_weights, v)

        return attn_output.squeeze(1)


class ImprovedDiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device
        self.n_actions = n_actions

        layers = []
        prev_size = input_shape
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, n_actions))

        self.model = nn.Sequential(*layers).to(self.device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs):
        return self.model(obs)

    def get_action(self, obs, deterministic=False):
        probs = self.get_output(obs)
        probs = torch.clamp(probs, min=1e-8, max=1 - 1e-8)

        if deterministic:
            return probs.cpu().numpy().argmax(), 0

        action = torch.multinomial(probs, 1, True)
        log_prob = torch.log(probs).gather(-1, action)

        return action.flatten().cpu(), log_prob.flatten().cpu()

    def get_output(self, obs):
        return self.softmax(self.forward(obs))

    def get_backprop_data(self, obs, acts):
        acts = acts.long()
        probs = self.get_output(obs)
        probs = torch.clamp(probs, min=1e-8, max=1 - 1e-8)

        log_probs = torch.log(probs)

        # Ensure acts has the correct shape for gathering
        if acts.dim() == 1:
            acts = acts.unsqueeze(-1)

        # Ensure log_probs and acts have compatible dimensions
        if log_probs.dim() == 3 and acts.dim() == 2:
            log_probs = log_probs.view(-1, log_probs.size(-1))
            acts = acts.view(-1, 1)

        action_log_probs = log_probs.gather(-1, acts)
        entropy = -(log_probs * probs).sum(dim=-1)

        return action_log_probs.to(self.device), entropy.to(self.device).mean()


class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device
        self.n_actions = n_actions

        self.attention = AttentionModule(input_shape, layer_sizes[0])
        self.improved_ff = ImprovedDiscreteFF(
            layer_sizes[0], n_actions, layer_sizes[1:], device
        )
        self.to(self.device)  # Move the entire model to the specified device

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        attn_output = self.attention(obs)
        return self.improved_ff(attn_output)

    def get_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        attn_output = self.attention(obs)
        return self.improved_ff.get_action(attn_output, deterministic)

    def get_output(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        attn_output = self.attention(obs)
        return self.improved_ff.get_output(attn_output)

    def get_backprop_data(self, obs, acts):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(acts, dtype=torch.long, device=self.device)
        attn_output = self.attention(obs)
        return self.improved_ff.get_backprop_data(attn_output, acts)
