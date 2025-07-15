import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=10, hidden_dims=[128, 64], dropout=0.3):
        super(DeepFM, self).__init__()

        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # ------------------------------
        # 1. Embedding layers : 각 feqture를 임베딩 벡터로 변환 (후에 FM 과 DNN에서 사용)
        # ------------------------------
        self.embedding = nn.ModuleList([
            nn.Embedding(field_dim, embed_dim) for field_dim in field_dims
        ])

        # ------------------------------
        # 2. Linear part (1st order term) : 각 feature의 단독 weight를 학습 
        # ------------------------------
        self.linear = nn.ModuleList([
            nn.Embedding(field_dim, 1) for field_dim in field_dims
        ])

        # ------------------------------
        # 3. Deep Neural Network (DNN) : FM에서 만든 각 feature 임베딩을 입력으로 받아서 학습
        #                               : 은닉층은 Relu 사용 
        # ------------------------------
        mlp_input_dim = self.num_fields * embed_dim
        mlp_layers = []
        for h_dim in hidden_dims:
            mlp_layers.append(nn.Linear(mlp_input_dim, h_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            mlp_input_dim = h_dim
        mlp_layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # ------------------------------
        # Linear term: 각 feature의 단독 weight를 학습하여 예측값에 더함
        # ------------------------------
        linear_terms = [emb(x[:, i]) for i, emb in enumerate(self.linear)]  # [(batch_size, 1), ...]
        linear_out = torch.sum(torch.cat(linear_terms, dim=1), dim=1, keepdim=True)  # (batch_size, 1)로 출력 

        # ------------------------------
        # FM part (2nd-order) : 각 feature 임베딩 벡터로 바꿔 FM 수식 사용 
        # ------------------------------
        embed_x = [emb(x[:, i]) for i, emb in enumerate(self.embedding)]  # [(batch_size, embed_dim), ...]
        embed_x = torch.stack(embed_x, dim=1)  # (batch_size, num_fields, embed_dim)

        sum_embed = torch.sum(embed_x, dim=1)  # (batch_size, embed_dim)
        sum_square = sum_embed ** 2
        square_sum = torch.sum(embed_x ** 2, dim=1)  # (batch_size, embed_dim)
        fm_out = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch_size, 1)

        # ------------------------------
        # DNN Term : 모든 임베딩 벡터를 flatten하여 입력 
        # ------------------------------
        dnn_input = embed_x.view(embed_x.size(0), -1)  # flatten: (batch_size, num_fields * embed_dim)
        dnn_out = self.mlp(dnn_input)  # (batch_size, 1)

        # ------------------------------
        # Final output
        # ------------------------------
        output = linear_out + fm_out + dnn_out  # (batch_size, 1)
        return torch.sigmoid(output)
