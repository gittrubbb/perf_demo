import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


# 1. 데이터 로딩 및 전처리

# user_features.csv: 사용자들의 연령, 성별, 계절, 사용빈도, 시간대, 옷 스타일, 색상 등 정보
# user_to_notes.csv: 각 사용자가 선호하는 향조
features_df = pd.read_csv('C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/data/user_features.csv')
notes_df = pd.read_csv('C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/data/user_to_notes.csv')

# user_id를 기준으로 두 데이터프레임 병합
df = pd.merge(features_df, notes_df, on="user_id")

# 사용자 입력 벡터 처리
X_raw = df[["age_group", "gender", "season", "freq_use", "time", "style", "color"]]
y_raw = df[["citrus", "floral", "woody", "musk", "spicy", "green", "sweet"]]

X_encoded = pd.get_dummies(X_raw)
X_tensor = torch.tensor(X_encoded.values, dtype=torch.float32)
# (500,28) shape


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


# 4. 향조 유사 사용자 프로파일 정의 및 latent 생성

# note_profiles: 향조별 선호할 법한 사용자 프로필 정보 -> AE의 Encoder에 입력될 수 있도록 마찬가지로 one-hot 인코딩
# -> latent 벡터로 변환 하여 향조 latent 생성 
note_profiles = [
    {"name": "citrus", "age_group": "20s", "gender": "F", "season": "summer", "freq_use": "daily", "time": "day", "style": "casual", "color": "white"},
    {"name": "woody",  "age_group": "40s", "gender": "M", "season": "autumn", "freq_use": "weekly", "time": "night", "style": "classic", "color": "brown"},
    {"name": "musk",   "age_group": "30s", "gender": "F", "season": "winter", "freq_use": "rarely", "time": "night", "style": "chic", "color": "black"},
    {"name": "floral", "age_group": "20s", "gender": "F", "season": "spring", "freq_use": "daily", "time": "day", "style": "romantic", "color": "beige"},
    {"name": "spicy",  "age_group": "30s", "gender": "M", "season": "autumn", "freq_use": "weekly", "time": "night", "style": "minimal", "color": "navy"},
    {"name": "green",  "age_group": "10s", "gender": "F", "season": "spring", "freq_use": "rarely", "time": "day", "style": "casual", "color": "green"},
    {"name": "sweet",  "age_group": "20s", "gender": "F", "season": "winter", "freq_use": "daily", "time": "night", "style": "romantic", "color": "pink"}
]

note_latents = []
note_names = []

for profile in note_profiles:
    df_note = pd.DataFrame([profile])
    note_names.append(df_note.pop("name").values[0])
    encoded = pd.get_dummies(df_note)
    for col in X_encoded.columns:
        if col not in encoded:
            encoded[col] = 0
    encoded = encoded[X_encoded.columns]  # column order match
    encoded = encoded.astype(float)
    tensor = torch.tensor(encoded.values, dtype=torch.float32)
    latent = model.encode(tensor)
    note_latents.append(latent)

note_latents_tensor = torch.cat(note_latents, dim=0)


# 5. 사용자 ↔ 향조 유사도 계산 & 추천
# 각각 사용자와 향조를 latent space에서 표현한 후, 코사인 유사도를 계산하여 추천 향조를 찾는다.
# Top 3 추천 향조를 argsosrt하여 출력.

model.eval()
with torch.no_grad():
    user_latent = model.encode(X_tensor)

sim_matrix = cosine_similarity(user_latent.detach().numpy(), note_latents_tensor.detach().numpy())
top_k = 3
recommendations = sim_matrix.argsort(axis=1)[:, -top_k:][:, ::-1]

for user_idx in range(5):
    rec_notes = [note_names[i] for i in recommendations[user_idx]]
    print(f"User {user_idx+1} → 추천 향조: {rec_notes}")