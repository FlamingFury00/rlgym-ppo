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


class ValueEstimator(nn.Module):
    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device

        self.attention = AttentionModule(input_shape, layer_sizes[0])

        layers = []
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)
        self.to(self.device)  # Move the entire model to the specified device

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        attn_output = self.attention(x)
        return self.model(attn_output)
