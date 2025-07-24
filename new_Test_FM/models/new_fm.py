import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    def __init__(self, field_dims, embedding_dim):
        '''
        field_dims: 각 field가 가질 수 있는 클래스 수의 리스트. 예: [4, 2, 16, 10, 10, 10, ..., 13]
        embedding_dim: 각 feature field의 embedding 차원 수
        '''
        super().__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(field_dim, embedding_dim) for field_dim in field_dims
        ])
        self.linear = nn.ModuleList([
            nn.Embedding(field_dim, 1) for field_dim in field_dims
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch_size, num_fields), 각 원소는 index (int)
        embed_x = torch.stack([emb(x[:, i]) for i, emb in enumerate(self.embedding)], dim=1)  # (B, F, D)
        linear_part = self.bias + sum([emb(x[:, i]) for i, emb in enumerate(self.linear)])   # (B, 1)

        sum_square = torch.sum(embed_x, dim=1) ** 2
        square_sum = torch.sum(embed_x ** 2, dim=1)
        interaction_part = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

        output = linear_part + interaction_part
        return torch.sigmoid(output)
