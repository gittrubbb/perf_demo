
# AI-based Perfume Recommendation Service: Perfuming 

🧠 설문 기반 사용자 프로필과 향조 정보를 바탕으로, 여러 추천 모델 비교 및 초기 개인화 향수 추천 파이프라인 실험용 Repository 
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

