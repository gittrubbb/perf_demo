from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import torch

def evaluate_binary_classification(y_true, y_pred_prob, threshold=0.5):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()

    y_pred = (y_pred_prob > threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    return acc, auc

def print_evaluation_results(y_true, y_pred_prob):
    acc, auc = evaluate_binary_classification(y_true, y_pred_prob)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")