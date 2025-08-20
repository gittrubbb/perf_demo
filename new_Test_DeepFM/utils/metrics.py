# metrics.py  (REPLACE WHOLE FILE)
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd

# -------------------- Core metrics --------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    uniq, first_idx, counts = np.unique(a[order], return_index=True, return_counts=True)
    for f, c in zip(first_idx, counts):
        if c > 1:
            idx = order[f:f+c]
            ranks[idx] = ranks[idx].mean()
    return ranks

def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def precision_recall_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int((y_true == 1).sum()))
    thresholds = y_score[order]
    return precision, recall, thresholds

def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = int((y_true == 1).sum())
    if pos == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve_points(y_true, y_score)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(ap)

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))

def f1_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    pred = (y_prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(f1), float(precision), float(recall)

# -------------------- Classification (per-label & aggregate) --------------------

def per_label_classification(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for note, g in df.groupby("note"):
        y = g["y_true"].to_numpy().astype(int)
        p = g["pred"].to_numpy().astype(float)
        if len(np.unique(y)) == 1:
            auc = float("nan")
            ap = float("nan")
        else:
            auc = roc_auc_binary(y, p)
            ap = average_precision(y, p)
        f1, prec, rec = f1_from_probs(y, p, threshold)
        bs = brier_score(y, p)
        rows.append((note, auc, ap, f1, prec, rec, bs, int(y.sum()), int((y==0).sum())))
    return pd.DataFrame(rows, columns=[
        "note","roc_auc","avg_precision","f1@0.5","precision@0.5","recall@0.5","brier","positives","negatives"
    ])

def macro_micro_aggregate(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    per_label = per_label_classification(df, threshold=threshold)
    macro_auc = float(np.nanmean(per_label["roc_auc"].to_numpy()))
    macro_ap  = float(np.nanmean(per_label["avg_precision"].to_numpy()))
    macro_f1  = float(np.mean(per_label["f1@0.5"].to_numpy()))

    y = df["y_true"].to_numpy().astype(int)
    p = df["pred"].to_numpy().astype(float)
    micro_auc = roc_auc_binary(y, p) if len(np.unique(y)) > 1 else float("nan")
    micro_ap  = average_precision(y, p)
    micro_f1, micro_prec, micro_rec = f1_from_probs(y, p, threshold)

    return {
        "macro_auc": macro_auc, "macro_ap": macro_ap, "macro_f1@0.5": macro_f1,
        "micro_auc": micro_auc, "micro_ap": micro_ap, "micro_f1@0.5": micro_f1,
        "micro_precision@0.5": micro_prec, "micro_recall@0.5": micro_rec
    }

# -------------------- Ranking per user --------------------

def _dcg(rel: np.ndarray) -> float:
    gains = (2 ** rel - 1)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum(gains * discounts))

def precision_at_k(rel: np.ndarray, k: int) -> float:
    k = max(1, k); return float(np.sum(rel[:k]) / k)

def recall_at_k(rel: np.ndarray, k: int) -> float:
    k = max(1, k); denom = max(1, int(np.sum(rel))); return float(np.sum(rel[:k]) / denom)

def ap_at_k(rel: np.ndarray, k: int) -> float:
    k = min(k, len(rel))
    if k <= 0: return 0.0
    ap, hit = 0.0, 0
    for i in range(k):
        if rel[i] > 0:
            hit += 1; ap += hit / (i + 1)
    denom = max(1, int(np.sum(rel)))
    return float(ap / denom)

def mrr_at_k(rel: np.ndarray, k: int) -> float:
    k = min(k, len(rel))
    for i in range(k):
        if rel[i] > 0: return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(rel: np.ndarray, k: int) -> float:
    k = min(k, len(rel))
    dcg = _dcg(rel[:k])
    ideal = -np.sort(-rel)[:k]
    idcg = _dcg(ideal)
    return float(dcg / idcg) if idcg > 0 else 0.0

def ranking_metrics(df: pd.DataFrame, ks: List[int]) -> Dict[str, float]:
    by_user = dict(tuple(df.groupby("user_id")))
    prec, rec, maps, mrrs, ndcgs = {k:[] for k in ks}, {k:[] for k in ks}, {k:[] for k in ks}, {k:[] for k in ks}, {k:[] for k in ks}
    for _, g in by_user.items():
        g = g.sort_values("pred", ascending=False)
        rel = g["y_true"].to_numpy().astype(int)
        for k in ks:
            prec[k].append(precision_at_k(rel, k))
            rec[k].append(recall_at_k(rel, k))
            maps[k].append(ap_at_k(rel, k))
            mrrs[k].append(mrr_at_k(rel, k))
            ndcgs[k].append(ndcg_at_k(rel, k))
    out = {}
    for k in ks:
        out[f"precision@{k}"] = float(np.mean(prec[k])) if prec[k] else 0.0
        out[f"recall@{k}"]   = float(np.mean(rec[k])) if rec[k] else 0.0
        out[f"map@{k}"]      = float(np.mean(maps[k])) if maps[k] else 0.0
        out[f"mrr@{k}"]      = float(np.mean(mrrs[k])) if mrrs[k] else 0.0
        out[f"ndcg@{k}"]     = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return out

# -------------------- Diversity --------------------

def label_coverage(topk_notes: List[str], all_labels: Iterable[str]) -> float:
    return float(len(set(topk_notes)) / max(1, len(set(all_labels))))

def distribution_entropy(labels: List[str]) -> float:
    cnt = Counter(labels); total = sum(cnt.values())
    if total == 0: return 0.0
    probs = np.array([c/total for c in cnt.values()], dtype=float)
    ent = -np.sum(probs * np.log2(np.maximum(probs, 1e-12)))
    return float(ent / np.log2(len(cnt))) if len(cnt) > 1 else 0.0  # [0,1] 정규화

def gini_index(labels: List[str]) -> float:
    cnt = Counter(labels); total = sum(cnt.values())
    if total == 0: return 0.0
    probs = np.array([c/total for c in cnt.values()], dtype=float)
    return float(1.0 - np.sum(probs**2))  # 높을수록 다양성↑

def diversity_from_df(df: pd.DataFrame, k: int = 1, all_labels: Iterable[str] = None) -> Dict[str, float]:
    picks = []
    for _, g in df.groupby("user_id"):
        picks.extend(g.sort_values("pred", ascending=False).head(k)["note"].tolist())
    if all_labels is None:
        all_labels = df["note"].unique().tolist()
    return {
        f"catalog_coverage@{k}": label_coverage(picks, all_labels),
        f"entropy@{k}": distribution_entropy(picks),
        f"gini@{k}": gini_index(picks),
    }

# -------------------- Report wrapper --------------------

def compute_all_metrics(df: pd.DataFrame, ks: List[int] = [1,3,5], threshold: float = 0.5) -> Dict:
    need = {"user_id","note","y_true","pred"}
    if need - set(df.columns):
        raise ValueError(f"Missing columns: {need - set(df.columns)}")
    per_label = per_label_classification(df, threshold)
    macro_micro = macro_micro_aggregate(df, threshold)
    rank = ranking_metrics(df, ks)
    diversity = {}
    for k in ks:
        diversity.update(diversity_from_df(df, k, all_labels=df["note"].unique().tolist()))
    return {
        "classification_per_label": per_label,
        "classification_macro_micro": macro_micro,
        "ranking": rank,
        "diversity": diversity,
    }

# -------------------- Helper: build eval DF --------------------

def build_eval_df_from_dataset_preds(user_ids, note_indices, y_true, pred_probs, note_label_encoder) -> pd.DataFrame:
    """
    input arrays must be aligned row-wise (same order used in model forward pass)
    returns DataFrame with columns: user_id, note, y_true, pred
    """
    df = pd.DataFrame({
        "user_id": user_ids,
        "note_idx": note_indices,
        "y_true": y_true.astype(float),
        "pred": pred_probs.astype(float),
    })
    df["note"] = note_label_encoder.inverse_transform(df["note_idx"].astype(int))
    # 혹시 중복이 있다면 안전하게 집계(대개는 1:1임)
    df = (df.groupby(["user_id","note"], as_index=False)
            .agg(y_true=("y_true","max"), pred=("pred","max")))
    return df[["user_id","note","y_true","pred"]]
