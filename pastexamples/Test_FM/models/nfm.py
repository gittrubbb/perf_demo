import torch
import torch.nn as nn

class NeuralFactorizationMachine(nn.Module):
    def __init__(self, input_dim, embedding_dim=10, hidden_dims=[64, 32], dropout=0.5):
        super(NeuralFactorizationMachine, self).__init__()

        self.embedding = nn.Parameter(torch.randn(input_dim, embedding_dim))
        self.linear = nn.Linear(input_dim, 1)

        mlp_layers = []
        input_size = embedding_dim
        for h_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_size, h_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_size = h_dim

        mlp_layers.append(nn.Linear(input_size, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        linear_output = self.linear(x)

        # Embedding을 직접 곱해서 latent representation 획득
        x_embed = torch.matmul(x, self.embedding)  # (batch_size, embed_dim)
        x_embed_sq = x_embed ** 2
        sq_embed = torch.matmul(x * x, self.embedding ** 2)
        bi_interaction = 0.5 * (x_embed_sq - sq_embed)

        deep_output = self.mlp(bi_interaction)
        output = linear_output + deep_output
        return torch.sigmoid(output)
