import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity


# 1. 데이터 로딩 및 전처리
# note_cols: 향조 컬럼 이름 리스트. (이후에 분리 목적)
note_cols = [
    "amber", "aquatic", "aromatic", "casual", "chypre", "citrus", "cozy", "floral",
    "fougere", "fruity", "gourmand", "green", "light_floral", "musk", "oriental",
    "powdery", "spicy", "white_floral", "woody"
]
# final_ae_input.csv: 사용자 정보와 향조 선호도를 포함한 통합 파일
df = pd.read_csv('C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/data/final_ae_input.csv', encoding='ISO-8859-1') 
 
# 사용자 입력 벡터 처리
# (1) 향조 제외한 사용자 정보 컬럼을 one-hot encoding
X_raw = df.drop(columns=["user_id"] + note_cols)
X_encoded = pd.get_dummies(X_raw)

# (2) 모든 값이 0~1 사이인지 확인
if not ((X_encoded.values >= 0).all() and (X_encoded.values <= 1).all()):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_encoded = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

# (500,28) shape의 tensor로 변환
X_tensor = torch.tensor(X_encoded.values, dtype=torch.float32) 



# 2. Autoencoder 정의
'''
입력: 28차원
-> Linear(28 -> 128) + ReLU
-> Linear(128 -> 32) + ReLU (latent space) (요약 정보)
-> Linear(32 -> 128) + ReLU
-> Linear(128 -> 28) + Sigmoid (복원한 출력) 
'''
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self, x):
        return self.encoder(x)


# 3. 학습

# criterion: BCELoss (Binary Cross Entropy Loss)
# optimizer: Adam(Momentum과 RMSProp의 결합: 진행하던 속도에 관성 주고, 적응적 학습률을 가짐)
model = Autoencoder(input_dim=X_tensor.shape[1], latent_dim=32)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, X_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# 4. 향조 프로파일 정의 

# note_profiles: final_ae_input.csv로 부터 각 향조에 대해 각 향조를 선호한 사용자들의 평균 특성을 자동으로 계산
# -> latent 벡터로 변환 하여 향조 latent 생성 
note_profiles_df = pd.read_csv("C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/data/final_ae_note_profiles.csv", index_col=0)

if "Unnamed: 0" in note_profiles_df.columns:
    note_profiles_df = note_profiles_df.drop(columns=["Unnamed: 0"])

# X_encoded와 공통 컬럼만 사용
common_cols = list(set(note_profiles_df.columns) & set(X_encoded.columns))
note_profiles_df = note_profiles_df[common_cols]

# 누락된 컬럼 0으로 채우기 (X_encoded 기준)
for col in X_encoded.columns:
    if col not in note_profiles_df.columns:
        note_profiles_df[col] = 0

# 최종적으로 학습 데이터와 동일한 feature 순서로 정렬
note_profiles_df = note_profiles_df[X_encoded.columns]
note_tensor = torch.tensor(note_profiles_df.values, dtype=torch.float32)

with torch.no_grad():
    user_latent = model.encode(X_tensor)
    note_latents_tensor = model.encode(note_tensor)


# 5. 사용자 ↔ 향조 유사도 계산 & 추천
# 각각 사용자와 향조를 latent space에서 표현한 후, 코사인 유사도를 계산하여 추천 향조를 찾는다.
# Top 3 추천 향조를 argsosrt하여 출력.

model.eval()
with torch.no_grad():
    user_latent = model.encode(X_tensor)

sim_matrix = cosine_similarity(user_latent.detach().numpy(), note_latents_tensor.detach().numpy())
top_k = 3
recommendations = sim_matrix.argsort(axis=1)[:, -top_k:][:, ::-1]

for user_idx in range(10):
    rec_notes = [note_cols[i] for i in recommendations[user_idx]]
    print(f"User {user_idx+1} → 추천 향조: {rec_notes}")