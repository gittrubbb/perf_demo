import pandas as pd
import torch
from torch.utils.data import Dataset

# PerfumeDataset 클래스 정의
class PerfumeDataset(Dataset):
    def __init__(self, user_feat_path: str, note_path: str):
        
        # 사용자/향조 데이터 불러오기
        features_df = pd.read_csv(user_feat_path)
        notes_df = pd.read_csv(note_path)

        # 향조 칼럼을 long 형식으로 변환 : 각 향조가 하나의 row로 표현되도록 변환
        note_cols = ["citrus", "floral", "woody", "musk", "spicy", "green", "sweet"]
        long_df = notes_df.melt(
            id_vars="user_id", value_vars=note_cols,
            var_name="note", value_name="liked"
        )

        # 사용자 특성과 향조 병합
        merged = pd.merge(features_df, long_df, on="user_id")

        # 입력 X, 타겟 y (liked: 1 or 0) 생성
        self.X_raw = merged[["age_group", "gender", "season", "freq_use", "time", "style", "color", "note"]]
        self.y = merged["liked"].values.astype('float32')

        # One-hot encoding 및 PyTorch tensor 변환
        self.X_encoded = pd.get_dummies(self.X_raw)
        self.X_tensor = torch.tensor(self.X_encoded.values, dtype=torch.float32)
        
        # 사용자 ID 저장
        self.user_ids = merged["user_id"].values

    # Dataset 전체 길이 반환
    def __len__(self):
        return len(self.X_tensor)

    # (x,y) 쌍 반환
    def __getitem__(self, idx):
        return self.X_tensor[idx], torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)

    # 입력 feature 차원 수 반환
    def get_input_dim(self):
        return self.X_tensor.shape[1]

    # 추천 결과를 사용자와 연결하기 위해서, 인코딩된 DataFrame 반환
    def get_encoded_df(self):
        return self.X_encoded.copy()
