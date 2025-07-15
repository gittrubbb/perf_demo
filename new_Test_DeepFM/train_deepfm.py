import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score

from models.deepfm import DeepFM
from data.perfume_dataset_deepfm import DeepFMDataset

def train_deepfm(user_path, note_path, embedding_dim, hidden_dims, dropout, epochs, batch_size):
    # 데이터셋 로드 
    dataset = DeepFMDataset(user_path, note_path)
    field_dims = dataset.get_field_dims()

    # 분할 
    test_ratio = 0.2
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 초기화
    model = DeepFM(field_dims, embed_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 평가: 확률값 0.5 기준으로 이진화 -> 정확도와 ROC-AUC 계산
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = model(X_batch)
            preds = (probs > 0.5).float()
            all_labels.extend(y_batch.squeeze().numpy())
            all_preds.extend(preds.squeeze().numpy())
            all_probs.extend(probs.squeeze().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")

    # 추천 결과 출력
    print("\n--- 향조 추천 결과 (상위 3개) ---")

    X_tensor = torch.tensor(dataset.X.values, dtype=torch.long)
    preds = torch.sigmoid(model(X_tensor)).detach().numpy().squeeze()

    df = pd.DataFrame({
        "user_id": dataset.user_ids,
        "note": dataset.notes,
        "pred": preds
        })
    print("user_id 순서 미리보기:", df["user_id"].unique()[:10])
    print("원본 user_id 순서 미리보기:", pd.read_csv(user_path)["user_id"].values[:10])

    # 예측 확률 높은 향조 추출
    df.sort_values("user_id", inplace=True)
    grouped = df.groupby("user_id")
    # for user_id, group in grouped:
    for i, (user_id, group) in enumerate(grouped):
        if i >= 100:
            break
        top_notes = group.sort_values("pred", ascending=False)["note"].head(3).tolist()
        print(f"User {user_id} → 추천 향조: {top_notes}")
