import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score

from models.new_fm import FactorizationMachine
from data.new_perfume_dataset import PerfumeDataset


def train_model(dataset_path_user, dataset_path_note, latent_dim=10, epochs=50, batch_size=64):
    # 데이터셋 로딩 및 분할 
    # input_dim: FM 모델의 입력 노드 수 (one-hot 인코딩된 feature의 수)
    dataset = PerfumeDataset(dataset_path_user, dataset_path_note)
    input_dim = dataset.get_input_dim()

    # 학습:테스트 = 80:20 비율로 분할
    test_ratio = 0.2
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # PyTorch DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 학습
    # Factorization Machine 모델 생성
    # Criterion: Binary Cross-Entropy Loss
    # Optimizer: Adam
    model = FactorizationMachine(input_dim, latent_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        avg_loss = running_loss / train_size
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 모델 평가
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = model(X_batch)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")

    # 향조 추천 출력
    print("\n--- 향조 추천 결과 (상위 3개) ---")
    
    encoded_df = dataset.get_encoded_df().reset_index(drop=True)
    encoded_df["note"] = dataset.X_raw["note"].values
    encoded_df["user_id"] = dataset.user_ids
    encoded_df["pred"] = torch.sigmoid(model(dataset.X_tensor)).detach().numpy()

   # 추가: user_id 순서 비교용
    print("\n--- User ID 순서 확인용 ---")
    print("원본 features_df user_id 순서 (일부):")
    print(pd.read_csv(dataset_path_user)["user_id"].unique()[:10])
    print("추천 결과 user_id 순서 (일부):")
    print(encoded_df["user_id"].unique()[:10])

    top_k = 3
    grouped = encoded_df.groupby("user_id")
    for user_id, group in grouped:
        top_notes = group.sort_values("pred", ascending=False)["note"].head(top_k).tolist()
        print(f"User {user_id} → 추천 향조: {top_notes}")

