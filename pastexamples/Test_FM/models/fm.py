import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    def __init__(self, input_dim: int, k: int):
        super(FactorizationMachine, self).__init__()
        self.linear = nn.Linear(input_dim, 1) # 선형항 (w0 + w1*x1 + ...)의 구현
        self.V = nn.Parameter(torch.randn(input_dim, k) * 0.01)

    def forward(self, x):
        linear_part = self.linear(x)

        # 2차 상호작용 항 계산
        # 0.5 * [(사용자 입력 * 임베딩벡터)의 각 임베딩차원의 제곱 - 직접 제곱 곱] 
        interaction_part = 0.5 * torch.sum(
            torch.pow(torch.matmul(x, self.V), 2) - torch.matmul(x * x, self.V * self.V),
            dim=1, keepdim=True
        )

        # 최종 출력 계산 (선형 part + 2차 상호작용 part)
        # Sigmoid 활성화 함수 적용 (0~1 사이의 확률로 변환)
        output = linear_part + interaction_part
        return torch.sigmoid(output)
