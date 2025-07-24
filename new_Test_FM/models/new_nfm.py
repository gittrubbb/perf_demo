import torch
import torch.nn as nn

class NeuralFactorizationMachine(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, hidden_dims=[64, 32], dropout=0.2):
        super(NeuralFactorizationMachine, self).__init__()
        
        self.num_fields = len(field_dims)
        self.embedding = nn.ModuleList([
            nn.Embedding(field_dim, embedding_dim, padding_idx=0) for field_dim in field_dims
        ])
        
        # Linear term 
        self.linear = nn.Linear(self.num_fields, 1)

        # MLP for bi-interaction layer 
        mlp_layers = []
        input_size = embedding_dim
        for dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_size, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout))
            input_size = dim

        mlp_layers.append(nn.Linear(input_size, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # x: (B, F) with integer indices
        embed_x = torch.stack([emb(x[:, i]) for i, emb in enumerate(self.embedding)], dim=1)  # (B, F, D)

        # Bi-Interaction Layer
        summed = torch.sum(embed_x, dim=1)  # (B, D)
        squared = embed_x ** 2
        summed_square = torch.sum(squared, dim=1)  # (B, D)
        bi_interaction = 0.5 * (summed ** 2 - summed_square)  # (B, D)

        # Pass through MLP
        deep_output = self.mlp(bi_interaction)  # (B, 1)

        # Linear term (assumes index-encoded categorical features)
        linear_input = x.float()  # if x is LongTensor, convert to float for linear layer
        linear_output = self.linear(linear_input)  # (B, 1)

        output = linear_output + deep_output
        return torch.sigmoid(output)
