import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

# PerfumeDataset 클래스 정의
class PerfumeDataset(Dataset):
    def __init__(self, user_feat_path: str, note_path: str):
        
        # 사용자/향조 데이터 불러오기
        user_df = pd.read_csv(user_feat_path)
        note_df = pd.read_csv(note_path)

        # 2. style, color 컬럼 전처리 (공백 제거 및 split)
        user_df["style"] = user_df["style"].fillna("").str.replace(" ", "").str.split(",")
        user_df["color"] = user_df["color"].fillna("").str.replace(" ", "").str.split(",")

        # 3. Multi-hot encoding (style, color)
        mlb_style = MultiLabelBinarizer()
        style_encoded = pd.DataFrame(mlb_style.fit_transform(user_df["style"]),
                                     columns=[f"style_{c}" for c in mlb_style.classes_])

        mlb_color = MultiLabelBinarizer()
        color_encoded = pd.DataFrame(mlb_color.fit_transform(user_df["color"]),
                                     columns=[f"color_{c}" for c in mlb_color.classes_])

        # 4. 단일 카테고리형 원핫 인코딩
        cat_cols = ["used_before", "gender", "age_group", "mbti", "freq_use", "time", "purpose", "price", "durability"]
        categorical_encoded = pd.get_dummies(user_df[cat_cols])

        # 5. 최종 사용자 feature 병합
        user_features = pd.concat([user_df[["user_id"]], categorical_encoded, style_encoded, color_encoded], axis=1)

        # 6. 향조 multi-hot 인코딩
        note_df["notes"] = note_df["notes"].fillna("").str.replace(" ", "").str.split(",")
        mlb_notes = MultiLabelBinarizer()
        note_encoded = pd.DataFrame(mlb_notes.fit_transform(note_df["notes"]), columns=mlb_notes.classes_)
        note_encoded["user_id"] = note_df["user_id"]

        # 7. user_id 기준 병합
        merged_df = pd.merge(user_features, note_encoded, on="user_id")

        # 8. 향조를 long-format으로 변환 (각 note별로 1 row씩)
        note_cols = mlb_notes.classes_.tolist()
        long_df = merged_df.melt(id_vars=user_features.columns.tolist(),
                                 value_vars=note_cols,
                                 var_name="note", value_name="liked")

        
        # 입력 X, 타겟 y (liked: 1 or 0) 생성
        self.X_raw = long_df.drop(columns=["user_id", "liked", "note"])
        self.X_raw["note"] = long_df["note"]  # note도 one-hot 처리 대상
        self.y = long_df["liked"].astype("float32").values

        # One-hot encoding 및 PyTorch tensor 변환
        self.X_encoded = pd.get_dummies(self.X_raw)
        self.X_encoded = self.X_encoded.astype(float)
        self.X_tensor = torch.tensor(self.X_encoded.values, dtype=torch.float32)
        
        # 사용자 ID 저장
        self.user_ids = long_df["user_id"].values

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
