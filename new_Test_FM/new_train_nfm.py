import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score

from models.new_nfm import NeuralFactorizationMachine
from data.new_perfume_dataset import PerfumeDataset

def train_nfm(dataset_path_user, dataset_path_note, embedding_dim=10, epochs=50, batch_size=64):
    dataset = PerfumeDataset(dataset_path_user, dataset_path_note)
    input_dim = dataset.get_input_dim()

    test_ratio = 0.2
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NeuralFactorizationMachine(input_dim, embedding_dim=embedding_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch+1) % 5 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = model(X_batch)
            preds = (probs > 0.5).float()
            all_labels.extend(y_batch.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")

    # 향조 추천 출력
    encoded_df = dataset.get_encoded_df().reset_index(drop=True)
    encoded_df["note"] = dataset.X_raw["note"].values
    encoded_df["user_id"] = dataset.user_ids
    encoded_df["pred"] = torch.sigmoid(model(dataset.X_tensor)).detach().numpy()

     # 향조 추천 획일화 문제점 확인용 추가 코드 
    print("\n--- 향조별 평균 예측 점수 ---")
    note_pred_mean = encoded_df.groupby("note")["pred"].mean().sort_values(ascending=False)
    print(note_pred_mean)


    top_k = 3
    grouped = encoded_df.groupby("user_id")
    print("\n--- 향조 추천 결과 (상위 3개) ---")
    for user_id, group in grouped:
        top_notes = group.sort_values("pred", ascending=False)["note"].head(top_k).tolist()
        print(f"User {user_id} → 추천 향조: {top_notes}")
