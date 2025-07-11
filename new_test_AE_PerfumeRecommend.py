import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 데이터 로딩
features_df = pd.read_csv("C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/data/pickply_user_features.csv")
notes_df = pd.read_csv("C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/data/pickply_user_to_notes.csv")

# 2. null 값 방지 및 공백 제거 및 리스트 분리 
features_df['style'] = features_df['style'].fillna('').str.replace(" ", "").str.split(",")
features_df['color'] = features_df['color'].fillna('').str.replace(" ", "").str.split(",")

mlb_style = MultiLabelBinarizer()
style_encoded = pd.DataFrame(mlb_style.fit_transform(features_df['style']), columns=[f"style_{c}" for c in mlb_style.classes_])

mlb_color = MultiLabelBinarizer()
color_encoded = pd.DataFrame(mlb_color.fit_transform(features_df['color']), columns=[f"color_{c}" for c in mlb_color.classes_])

# 3. 단일 카테고리형 처리
categorical_cols = ["used_before", "gender", "age_group", "mbti", "freq_use", "time", "purpose", "price", "durability"]
categorical_encoded = pd.get_dummies(features_df[categorical_cols])

# 4. 전체 feature 통합
final_features = pd.concat([features_df[["user_id"]], categorical_encoded, style_encoded, color_encoded], axis=1)

# 5. 노트 multi-hot 인코딩
notes_df['notes'] = notes_df['notes'].fillna('').str.replace(" ", "").str.split(",")
mlb_note = MultiLabelBinarizer()
note_encoded = pd.DataFrame(mlb_note.fit_transform(notes_df['notes']), columns=mlb_note.classes_)
note_encoded["user_id"] = notes_df["user_id"]

# 6. 병합
merged_df = pd.merge(final_features, note_encoded, on="user_id")
note_cols = mlb_note.classes_.tolist()

# 7. Autoencoder 입력 생성
X_raw = merged_df.drop(columns=["user_id"] + note_cols)
X_raw = X_raw.astype(float)
X_tensor = torch.tensor(X_raw.values, dtype=torch.float32)

# 8. Autoencoder 정의
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def encode(self, x):
        return self.encoder(x)

# 9. 학습
model = Autoencoder(input_dim=X_tensor.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, X_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 10. 향조 프로필 → 평균 latent 생성
model.eval()
with torch.no_grad():
    user_latents = model.encode(X_tensor)
    note_latents = []
    for note in note_cols:
        indices = merged_df[note] == 1
        note_latents.append(user_latents[indices].mean(dim=0) if indices.sum() > 0 else torch.zeros(user_latents.shape[1]))
    note_latents_tensor = torch.stack(note_latents)

# 11. 코사인 유사도 계산
sim_matrix = cosine_similarity(user_latents.numpy(), note_latents_tensor.numpy())
top_k = 3
recommendations = sim_matrix.argsort(axis=1)[:, -top_k:][:, ::-1]

# 12. 결과 출력
for i, rec in enumerate(recommendations[:10]):
    top_notes = [note_cols[j] for j in rec]
    print(f"User {i+1} → 추천 향조: {top_notes}")
    
