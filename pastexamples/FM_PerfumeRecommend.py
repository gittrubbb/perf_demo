import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# 1. 데이터 로딩 및 전처리

# user_features.csv: 사용자들의 연령, 성별, 계절, 사용빈도, 시간대, 옷 스타일, 색상 등 정보
# user_to_notes.csv: 각 사용자가 선호하는 향조
features_df = pd.read_csv('C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/data/user_features.csv')
notes_df = pd.read_csv('C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/data/user_to_notes.csv')

# 여러 개의 향조 칼럼을, 길게 변환 -> melt 형식으로 변환 (multi-label -> binary classification 데이터)
# 사용자 ID를 기준으로 두 데이터프레임 병합
note_cols = ["citrus", "floral", "woody", "musk", "spicy", "green", "sweet"]
long_df = notes_df.melt(id_vars="user_id", value_vars=note_cols, var_name="note", value_name="liked")
merged = pd.merge(features_df, long_df, on="user_id")

# One-hot encoding for categorical features + note
# 출력 값은 Liked (0 or 1)
X_raw = merged[["age_group", "gender", "season", "freq_use", "time", "style", "color", "note"]]
y = merged["liked"]

X_encoded = pd.get_dummies(X_raw)

# Pandas -> PyTorch tensor 변환
X_tensor = torch.tensor(X_encoded.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# 2. Factorization Machine 모델 정의

# input_dim: 입력 벡터의 차원
# k: latent factor의 차원 (임베딩 차원)
class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, k):
        super(FactorizationMachine, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.V = nn.Parameter(torch.randn(input_dim, k))  # embedding matrix
# Linear: 1차 항 (w_0 + w_i * x_i)
# V: k차원 임베딩 행렬 (각 feature에 대한 latent factor)

# FM 수식의 구현
    def forward(self, x):
        linear_part = self.linear(x)

        # Pairwise interactions (FM core)
        interaction_part = 0.5 * torch.sum(
            torch.pow(torch.matmul(x, self.V), 2) - torch.matmul(x * x, self.V * self.V), dim=1, keepdim=True
        )
        output = linear_part + interaction_part
        return torch.sigmoid(output)


# 3. 학습

# criterion: BCELoss (Binary Cross Entropy Loss)
# optimizer: Adam(Momentum과 RMSProp의 결합: 진행하던 속도에 관성 주고, 적응적 학습률을 가짐)
input_dim = X_train.shape[1]
latent_dim = 10
model = FactorizationMachine(input_dim, latent_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# 4. 평가

model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test)
    y_pred = (y_pred_prob > 0.5).float()

acc = accuracy_score(y_test.numpy(), y_pred.numpy())
auc = roc_auc_score(y_test.numpy(), y_pred_prob.numpy())
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test ROC-AUC: {auc:.4f}")

# ---------------------------
# 5. 사용자별 향조 추천 출력
# ---------------------------

# X_test에 해당하는 merged DataFrame 행 추출
X_train_idx, X_test_idx, _, _ = train_test_split(
    merged.index, merged.index, test_size=0.2, random_state=42
)
merged_test = merged.iloc[X_test_idx].copy().reset_index(drop=True)

# 예측 결과 추가
merged_test["pred"] = y_pred_prob.detach().numpy()

# 사용자별 향조 예측 정렬 및 top-K 추천 출력
top_k = 3
user_grouped = merged_test.groupby("user_id")

print("\n--- 향조 추천 결과 (상위 3개) ---")
for user_id, group in user_grouped:
    top_notes = group.sort_values("pred", ascending=False)["note"].head(top_k).tolist()
    print(f"User {user_id} → 추천 향조: {top_notes}")
