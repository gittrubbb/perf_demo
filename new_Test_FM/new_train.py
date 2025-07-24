import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from models.new_fm import FactorizationMachine
from data.perfume_dataset_FM import PerfumeDataset


def train_model(dataset_path_user, dataset_path_note, embedding_dim, epochs, batch_size):
    # 데이터셋 로딩 및 분할 
    # input_dim: FM 모델의 입력 노드 수 (one-hot 인코딩된 feature의 수)
    dataset = PerfumeDataset(dataset_path_user, dataset_path_note)
    field_dims = dataset.get_input_dim()

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
    model = FactorizationMachine(field_dims, embedding_dim=embedding_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 학습 루프
    num_epochs = 100
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(X_batch) 
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 

        if (epoch+1) % 5 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 모델 평가
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
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
    X_tensor = torch.tensor(dataset.X.values, dtype=torch.long)
    preds = torch.sigmoid(model(X_tensor.long())).detach().numpy().squeeze()
    
    note_le = dataset.get_label_encoders()["note"]

    df = pd.DataFrame({
        "user_id": dataset.user_ids,
        "note": dataset.notes,
        "pred": preds
    })

    # 향조별 평균 예측 점수 출력
    print("\n--- 향조별 평균 예측 점수 ---")
    note_pred_mean = df.groupby("note")["pred"].mean().sort_values(ascending=False)
    print(note_pred_mean)
    
    note_counts = df["note"].value_counts()
    print("\n --향조별 샘플 수 -- ")
    print(note_counts)

    # 사용자별 향조 예측 점수 추출
    print("\n--- 사용자별 향조 예측 확률 ---")
    grouped = df.groupby("user_id")
    for i, (user_id, group) in enumerate(grouped):
        print(f"\n[User {user_id}] 향조별 예측 점수:")
        for _, row in group.sort_values("pred", ascending=False).iterrows():
            print(f"  - {row['note']}: {row['pred']:.4f}")
        if i >= 5:
            print("... (이후 생략) ...")
            break

    # 사용자별 향조 top 3 추천
    df.sort_values("user_id", inplace=True)
    grouped = df.groupby("user_id")
    for i, (user_id, group) in enumerate(grouped):
        if i >= 50:
            break
        top_notes = group.sort_values("pred", ascending=False)["note"].head(3).tolist()
        print(f"User {user_id} → 추천 향조: {top_notes}")
