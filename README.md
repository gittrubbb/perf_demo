현재 평가 지표 수치 현황 
1. Macro/Micro 분류 지표

macro_auc = 0.6926 / micro_auc = 0.7298
→ 무작위(0.5) 대비 꽤 높은 수준입니다. 모델이 사용자–향조 매칭에서 신호를 분명히 잡고 있어요.
→ 다만 0.7대 초반이라 “준수한 성능, 하지만 아주 강력한 분류기는 아님” 정도로 해석합니다.

macro_ap = 0.3726 / micro_ap = 0.4045
→ 불균형 데이터에서 AP는 더 까다로운 지표인데, 0.37~0.40 수준이면 baseline 대비 의미 있는 신호가 있고, 개선余地도 큽니다.

F1 (macro=0.3384, micro=0.3436)
→ 낮은 편입니다.

Precision@0.5 = 0.2075 (정밀도 낮음)

Recall@0.5 = 1.0 (재현율 100%)
즉, threshold=0.5에서는 모든 positive를 잡아내지만, 음성까지 많이 긁어오는 과다예측 상태입니다.
→ Threshold calibration 또는 per-label thresholding을 하면 Precision-Recall 균형이 훨씬 나아질 수 있습니다.

2. Ranking @K

Precision@1 = 0.4248 / Recall@1 = 0.1942
→ Top-1 예측의 적중률은 42%, 평균적으로 실제 positive의 19%만 잡습니다.
→ Top-1 추천은 아직 약간 부족.

Recall@3 = 0.4594 / MAP@3 = 0.3525 / NDCG@3 = 0.4650
→ Top-3 안에서는 절반 가까이 실제 positive를 포함시키고 있고, NDCG도 0.46으로 괜찮습니다.
→ 즉, Top-3 추천 기준으로는 꽤 쓸 만하다고 평가할 수 있습니다.

MRR@3 = 0.5615
→ 첫 번째 relevant item이 평균적으로 2위 안팎에 위치한다는 의미 → 꽤 양호합니다.

3. 다양성 (Diversity)

Coverage@1 = 83%
→ Top-1만 봐도 전체 향조의 83%가 실제 추천으로 쓰임 → 특정 라벨에 쏠림이 심하지 않다는 의미.

Coverage@3 = 100%
→ Top-3까지 합치면 전체 향조가 모두 커버됨 → 추천 다양성은 훌륭함.

Entropy, Gini도 K가 커질수록 상승 → 라벨 분포가 고르게 확산되고 있다는 긍정적 신호.

즉, 균등화/샘플링 전략이 실제로 다양성에 크게 기여하고 있음을 확인할 수 있습니다.

4. Per-Label 성능

강한 라벨:

floral (ROC-AUC=0.74, AP=0.46), light_floral (AUC=0.78, AP=0.41), fruity (AUC=0.72, AP=0.34) → 상대적으로 잘 학습.

약한 라벨:

musk/citrus/aquatic/green 등은 AUC가 0.620.68, AP도 0.270.43 수준 → 개선 필요.

공통 패턴:

Recall=1.0, Precision 낮음 → threshold 조정 필요성을 다시 보여줌.



# 향수 추천 시스템 (Perfume Recommendation System)

🧠 설문 기반 사용자 맞춤 향수 추천 시스템
📂 사용 모델: AutoEncoder, Factorization Machine, Neural FM, DeepFM

## 🧠 사용한 모델

| 모델 | 설명 | 평가 |
|-------------------------------|----------------------------------------------------------------|----------------------------------------|
|          AutoEncoder          | 사용자 특성 → latent 벡터로 압축 후 복원, 향조 유사도 기반 추천 | 콘텐츠 기반 추천, 다양한 향조 추천 가능 |
|   Factorization Machine (FM)  |              사용자와 향조의 이차 상호작용 모델링               |         성능 우수 (Accuracy ↑)         |
|        Neural FM (NFM)        |                    FM의 이차항을 MLP에 입력                    |    향조 다양성 다소 부족 (조정 필요)    |
|            DeepFM             |              FM + DNN (고차원 임베딩 레이어)                   |                    -                   |

## 📁 프로젝트 구조
perf_demo/

├── AE/
  
  └── new_test_AE_PerfumeRecommend.py     # AutoEncoder 구현
  
  ├── new_Test_FM/

      ├── new_train.py                    # FM 학습 및 평가
  
      ├── new_train_nfm.py                # NFM 학습 및 평가
  
      ├── models/
  
        ├── new_fm.py                     # FM 모델 정의
        └── new_nfm.py                    # NFM 모델 정의
  
      ├── data/
        └── new_perfume_dataset.py        # 사용자 + 향조 전처리
  
      └── new_main.py                     # 실행 스크립트
      └── new_main_nfm.py

  ├── new_Test_DeepFM/

      ├── train_deepfm.py                 # DeepFM 학습 및 평가
  
      ├── main_deepfm.py                  # 실행 스크립트 
  
      ├── models/
  
        ├── deepfm.py                     # DeepFM 모델 정의
  
      ├── data/
        └── new_perfume_dataset.py        # 사용자 + 향조 전처리
  
  ├── data_files/                         # 설문 응답 및 매핑된 향조

  ├── requirements.txt                    # 패키지 리스트

  └── README.md

