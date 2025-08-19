import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import ast
import random 
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# PerfumeDataset 클래스 정의
class PerfumeDataset(Dataset):
    def __init__(self, data_path: str):
        
        # 다중 선택(multi-label) 항목에 대한 고정 slot 수 설정 (ex) style에서 4개의 답변을 선택해도, 3개만 반영
        self.fixed_slots = { "purpose": 2, "fashionstyle": 3, "prefercolor": 3, "perfume_category" : 8 } 
        PAD_TOKEN = "PAD"
        
        # 사용자/향조 데이터 불러오기
        df = pd.read_csv(data_path)
        
        # Multi-label 항목들의 처리: fixed slot 갯수에 맞추어, 부족한 항목은 PAD로 채움. 
        # (ex) 3개의 fixed slot인 style에서 2개의 답변만 선택한 사용자에겐, 1칸은 PAD로 할당. 
        def process_multilabel_column(df, col, slots):
            df[col] = df[col].fillna("").astype(str).str.replace(" ", "").str.split(",")
            df[col] = df[col].apply(lambda x: x[:slots] + [PAD_TOKEN] * (slots - len(x)))
            return df 
        
        # 문자열 -> 리스트 변환 및 쉼표 분리       
        def safe_literal_eval(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) else []
            except:
                return [x] if isinstance(x, str) else []
        
        def clean_and_split_notes(x):
            if isinstance(x, str):
                x = x.replace(" ", "")  # 공백 제거
                return x.split(",")     # 쉼표 기준 분리
            elif isinstance(x, list):   # 리스트 안에 ','가 있는 문자열이 있는 경우 다시 나눠줌
                new_list = []
                for item in x:
                    if isinstance(item,str) and "," in item:
                        new_list.extend(item.replace(" ", "").split(","))
                    else:
                        new_list.append(item.strip() if isinstance(item, str) else item)
                return new_list
            else:
                return []
            
        for field in ["purpose", "fashionstyle", "prefercolor"]:
            df = process_multilabel_column(df, field, self.fixed_slots[field])
                
        df["perfume_category"] = df["perfume_category"].fillna("[]").apply(safe_literal_eval)
        df["perfume_category"] = df["perfume_category"].fillna("").apply(clean_and_split_notes)
        
        # 향조 분포 확인용 출력 (7/22)
        all_notes_flat = [note for row in df["perfume_category"] for note in row if isinstance(note, str)]
        from collections import Counter
        note_counter = Counter(all_notes_flat)
        print("\n[향조 문자열 분포 상위 30개] - 1")
        print(note_counter.most_common(30))

        # 향조도 fixed slot 길이에 맞추어 빈칸엔 PAD 처리 
        df["perfume_category"] = df["perfume_category"].apply(
            lambda x: x[:self.fixed_slots["perfume_category"]] + [PAD_TOKEN] * (self.fixed_slots["perfume_category"] - len(x))
        )
        
        # Positive 샘플 생성 (사용자가 선택한 향조)
        positive_rows = []
        for _, row in df.iterrows():
            for note in row["perfume_category"]:
                if note != PAD_TOKEN:
                    positive_rows.append({"user_id": row["user_id"], "note": note, "liked": 1})
        positive_df = pd.DataFrame(positive_rows)
        
        # 8/19 positive 수 줄이기 
        # === (A) 현재 분포 확인 ===
        from collections import Counter
        cur_cnt = Counter(positive_df["note"])
        print("[Before] positive count per note (top 30):", cur_cnt.most_common(30))

        # === (B) 노트별 타깃 개수 설정 ===
        # 주신 계획을 그대로 반영 (값이 애매하게 적힌 부분은 명시값 우선)
        target_counts = {
            "woody": 703,      # 852 -> 703 (–149)  *주신 -150 의도치 반영
            "musk": 571,       # 720 -> 571 (–149)
            "citrus": 565,     # 715 -> 565 (–150)
            "aquatic": 520,    # 570 -> 520 (–50)
            "green": 435,      # 485 -> 435 (–50)
            "floral": 411,     # 461 -> 411 (–50)
    # 아래는 변화 없음(그대로 두기) — 필요시 여기도 타깃값 채우면 됨
    # "casual": 386, "light_floral": 351, "powdery": 345, "fruity": 310,
    # "aromatic": 286, "cozy": 212, "white_floral": 197, "chypre": 194,
    # "fougere": 176, "amber": 173, "gourmand": 123,
        }

        ALLOW_OVERSAMPLE = True   # oriental처럼 늘리고 싶은 클래스가 있을 때 True
        RNG = np.random.default_rng(42)

        # === (C) 노트별 다운/오버샘플 ===
        kept_chunks = []    
        all_notes = positive_df["note"].unique().tolist()

        for note in all_notes:
            df_n = positive_df[positive_df["note"] == note]
            cur = len(df_n)
            tgt = target_counts.get(note, cur)  # 타깃 미설정이면 기존 개수 유지

            if tgt < cur:
                idx = RNG.choice(df_n.index.values, size=tgt, replace=False)
                kept = df_n.loc[idx]

            else:
                kept = df_n

            kept_chunks.append(kept)

        positive_df_bal = pd.concat(kept_chunks, ignore_index=True)

# === (D) 사후 분포 확인 ===
        new_cnt = Counter(positive_df_bal["note"])
        print("[After ] positive count per note (top 30):", new_cnt.most_common(30))

# 이후 파이프라인에서 positive_df 대신 positive_df_bal 사용
# ex) all_df = pd.concat([positive_df_bal, negative_df], ignore_index=True)


        all_notes = set(positive_df_bal["note"].unique()) - {PAD_TOKEN}
        
        
        negative_sample_ratio = 4
        negative_rows = []
        
        for uid, user_notes in df[["user_id", "perfume_category"]].values:
            unselected = all_notes - set(user_notes)
            sampled = random.sample(unselected, min(len(unselected), negative_sample_ratio * len(user_notes)))
            for note in sampled:
                negative_rows.append({"user_id": uid, "note": note, "liked": 0})
        negative_df = pd.DataFrame(negative_rows)
        
        '''
        # 향조별 등장 빈도 계산
        note_counter = Counter([note for row in df["perfume_category"] for note in row if note != PAD_TOKEN])
        total_count = sum(note_counter.values())

        # 향조별 등장 빈도 기반 샘플링 확률 계산
        note_sampling_prob = {note: count / total_count for note, count in note_counter.items()}

        # Negative sample 생성 (확률 기반)
        negative_sample_ratio = 4
        negative_rows = []

        for uid, user_notes in df[["user_id", "perfume_category"]].values:
            user_notes_set = set(user_notes)
            candidates = list(all_notes - user_notes_set)
    
            if not candidates:
                continue

            # 각 후보 노트에 대해 확률 추출
            weights = np.array([note_sampling_prob[n] for n in candidates])
            weights /= weights.sum()  # 정규화

            sample_size = min(len(candidates), negative_sample_ratio * len(user_notes_set))
            sampled_negatives = np.random.choice(candidates, size=sample_size, replace=False, p=weights)

            for note in sampled_negatives:
                negative_rows.append({"user_id": uid, "note": note, "liked": 0})
        negative_df = pd.DataFrame(negative_rows)
        '''
        # Positive + Negative 결합 
        long_df = pd.concat([positive_df_bal, negative_df], ignore_index=True)
        merged = pd.merge(df, long_df, on="user_id")
        
        # Fixed slot 구조로 feature 필드 전개 (flatten)
        self.feature_cols = (
            ["age_group", "gender", "mbti"] +
            [f"purpose_{i+1}" for i in range(self.fixed_slots["purpose"])] +
            [f"fashionstyle_{i+1}" for i in range(self.fixed_slots["fashionstyle"])] +
            [f"prefercolor_{i+1}" for i in range(self.fixed_slots["prefercolor"])] +
            ["note"]
        )

        # 여러개 필드 -> 컬럼으로 flatten
        for field, slots in self.fixed_slots.items():
            if field != "perfume_category":  # already exploded
                col_data = merged[field].tolist()
                for i in range(slots):
                    merged[f"{field}_{i+1}"] = [row[i] for row in col_data]
        
        # Feature 구분력 확인 (7/22)            
        print("\n[Feature 구분력 확인]")
        for col in [f"purpose_{i+1}" for i in range(2)] + \
                    [f"fashionstyle_{i+1}" for i in range(3)] + \
                    [f"prefercolor_{i+1}" for i in range(3)]:
            print(f"{col} value counts:\n{merged[col].value_counts()}\n")

        merged["note"] = merged["note"].astype(str)

        # Label encoding
        self.label_encoders = {}
        self.X = pd.DataFrame()
        for col in self.feature_cols:
            le = LabelEncoder()
            le.fit(["PAD"] + list(set(merged[col].astype(str)) - {"PAD"}))  # PAD 먼저
            self.X[col] = le.fit_transform(merged[col].astype(str))
            self.label_encoders[col] = le

        self.y = merged["liked"].astype("float32").values
        self.user_ids = merged["user_id"].values
        self.notes = merged["note"].values        
        

    # Dataset 전체 길이 반환
    def __len__(self):
        return len(self.X)

    # (x,y) 쌍 반환
    def __getitem__(self, idx):
        x = torch.tensor(self.X.iloc[idx].values, dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
        return x, y

    # 입력 feature 차원 수 반환
    def get_input_dim(self):
        return [len(le.classes_) for le in self.label_encoders.values()]

    def get_label_encoders(self):
        return self.label_encoders
