import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class DeepFMDataset(Dataset):
    def __init__(self, user_feat_path: str, note_path: str):
        # 데이터 로딩
        features_df = pd.read_csv(user_feat_path)
        notes_df = pd.read_csv(note_path)

        # 모델에서 추천하고자 하는 19종의 향조목록 
        note_cols = [
            "citrus", "floral", "woody", "oriental", "musk", "aquatic", "green",
            "gourmand", "powdery", "fruity", "aromatic", "chypre", "fougere", "amber",
            "spicy", "light_floral", "white_floral", "casual", "cozy"
            ]
        all_notes = set(note_cols)
        
        # notes 컬럼을 리스트로 변환 (", " 또는 "," 기준)
        notes_df["notes"] = notes_df["notes"].str.replace(" ", "")  # 공백 제거
        notes_df["notes"] = notes_df["notes"].str.split(",")        # 쉼표로 리스트화
        
        # positive 샘플 생성: user_id가 선택한 향조들을 "liked"로 표시
        positive_df = notes_df.explode("notes").rename(columns={"notes": "note"})
        positive_df["liked"] = 1

        # negative 샘플 생성: 선택하지 않은 향조들은 "not liked"(0)로 표시
        negatives = []
        for uid, notes in notes_df[["user_id", "notes"]].values:
            selected = set(notes)
            unselected = all_notes - selected
            for note in unselected:
                negatives.append({"user_id": uid, "note": note, "liked": 0})

        negative_df = pd.DataFrame(negatives)
        
        # 총 sample수는 user 수 * 19 (모든 향조) 
        # positive + negative 결합
        long_df = pd.concat([positive_df, negative_df], ignore_index=True)
        
        # 병합
        merged = pd.merge(features_df, long_df, on="user_id")

        # 사용할 feature 선택
        self.feature_cols = [
                "used_before", "age_group", "gender", "time", 
                "freq_use", "style", "color", "mbti", "purpose", 
                "price", "durability", "note"]
        self.label_col = "liked"

        # Label Encoding
        self.label_encoders = {}
        self.X = pd.DataFrame()
        # 범주형 feature를 숫자 인덱스로 변환
        for col in self.feature_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(merged[col].astype(str))
            self.label_encoders[col] = le

        self.y = merged[self.label_col].values.astype("float32")
        
        self.notes = merged["note"].values  # 추천 결과 매핑용 원본 note 문자열 저장
        self.user_ids = merged["user_id"].values  # 추천 결과 매핑용 user_id 저장


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X.iloc[idx].values, dtype=torch.long)  # 각 feature는 index
        y = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
        return x, y

    def get_field_dims(self):
        return [len(le.classes_) for le in self.label_encoders.values()]

    def get_label_encoders(self):
        return self.label_encoders
    