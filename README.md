
# AI-based Perfume Recommendation Service: Perfuming 

This repository contains experimental implementations of several recommendation models explored during the development of an AI-based personalized perfume recommendation system. The experiments focus on modeling user preference from structured profile data and survey responses, with particular attention to the practical challenges of real-world recommendation systems such as limited data, class imbalance, and noisy user feedback.

These experiments were conducted as part of the early-stage development of a personalized fragrance recommendation service later deployed in a kiosk-based interactive system.

## Motivation 
Personalized recommendation systems often rely on large-scale interaction data. However, in real-world deployment scenarios—especially for emerging services—initial datasets are typically small, noisy, and highly imbalanced.

During the development of an AI-based perfume recommendation platform, we encountered several practical challenges:

Limited user interaction data during early-stage service deployment

Noisy and subjective survey responses

Imbalanced preference distributions across fragrance notes

Difficulty maintaining stable prediction performance under evolving user behavior

This repository was created to experiment with different recommendation models and analyze their behavior under such constraints before integrating them into a production system.

## Problem Setting 
The goal of this project is to model user fragrance preferences based on structured user profiles and survey-derived features.

## Output Target
The models aim to predict user preference for fragrance components (notes) or provide personalized perfume recommendations based on the learned feature interactions.

## Models Explored 
Several recommendation models were implemented and compared to understand how different architectures handle sparse and structured user feature data.

1. AutoEncoder (AE)
A neural network–based collaborative filtering approach used as a baseline for reconstructing user preference representations.

2. Factorization Machine (FM)
A model designed to capture pairwise feature interactions efficiently in sparse feature spaces.

3. Neural Factorization Machine (NFM)
An extension of FM that uses neural networks to model non-linear feature interactions.

4. DeepFM
A hybrid architecture that combines the strengths of factorization machines and deep neural networks, enabling both low-order and high-order feature interaction learning.

DeepFM was selected as the primary model due to its ability to simultaneously model explicit feature interactions and deeper non-linear relationships within structured user profiles.

## Repository Sturcture 
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

## Scope of this repository 
This repository contains the model experimentation and development stage of the recommendation system.

The following components are not included in this repository:

1. production kiosk application
2. user interface and interaction logic
3. hardware integration with digital scent devices
4. deployment pipelines

Those components were developed separately as part of the full system integration

## Limitations 
Several limitations were identified during experimentation:

1. limited dataset size during early-stage service development
2. noisy and subjective survey-based preference signals
3. imbalanced distribution of fragrance note preferences
4. potential distribution shifts as new user data is collected

These issues highlight the challenges of deploying recommendation systems in real-world environments where data quality and quantity evolve over time.

## Author 
Jihong Yi
B.S. Electronics Engineering
Chungnam National University
