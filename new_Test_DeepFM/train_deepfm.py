import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from models.deepfm import DeepFM
from data.perfume_dataset_deepfm import PerfumeDataset

# DeepFM 학습 및 평가 함수 정의
def train_deepfm(data_path, embedding_dim, hidden_dims, dropout, epochs, batch_size):
    # 데이터셋 로드 및 feature 정보 추출  
    dataset = PerfumeDataset(data_path)
    field_dims = dataset.get_input_dim() # 각 field의 클래스 갯수 리스트 반환 

    # 학습/검증 세트 분할 
    test_ratio = 0.2
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoader 구성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 정의
    model = DeepFM(field_dims, embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)
    criterion = nn.BCELoss() # 출력: 0~1의 확률
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 학습 루프
    num_epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
            preds = (probs > 0.5).float() #0.5 기준으로 이진 분류 
            all_labels.extend(y_batch.squeeze().numpy())
            all_preds.extend(preds.squeeze().numpy())
            all_probs.extend(probs.squeeze().numpy())

    # 성능 지표 출력 
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")
    

    # 추천 결과 출력 (테스트용)
    print("\n--- 향조 추천 결과 (상위 3개) ---")

    X_tensor = torch.tensor(dataset.X.values, dtype=torch.long)
    preds = torch.sigmoid(model(X_tensor)).detach().numpy().squeeze() #모든 샘플에 대한 예측 

    note_le = dataset.get_label_encoders()["note"]

    # 예측 결과 DataFrame 생성
    df = pd.DataFrame({
        "user_id": dataset.user_ids,
        "note_idx": dataset.X["note"],
        "pred": preds
    })

    # LabelEncoder를 통해 index → 문자열로 복원
    df["note"] = note_le.inverse_transform(df["note_idx"])
    df = df.drop_duplicates(subset=["user_id", "note"])
    
     # 향조 추천 획일화 문제점 확인용 추가 코드 
    print("\n--- 향조별 평균 예측 점수 ---")
    note_pred_mean = df.groupby("note")["pred"].mean().sort_values(ascending=False)
    print(note_pred_mean)
    
    # 향조별 샘플 수 확인 
    note_counts = df["note"].value_counts()
    print("\n--- 향조별 샘플 수 ---")
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
    
    # 예측 확률 높은 향조 추출
    df.sort_values("user_id", inplace=True)
    grouped = df.groupby("user_id")
    for i, (user_id, group) in enumerate(grouped):
        if i >= 100:
            break
        top_notes = group.sort_values("pred", ascending=False)["note"].head(2).tolist()
        print(f"User {user_id} → 추천 향조: {top_notes}")
