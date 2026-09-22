from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

def empirical_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))

def pb_count(scores: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return (scores[:, None] * probs).sum(axis=0)

def pb_variance(scores: np.ndarray, probs: np.ndarray) -> np.ndarray:
    w = scores[:, None] * probs
    return (w * (1.0 - w)).sum(axis=0)

def pb_covariance(scores: np.ndarray, probs: np.ndarray) -> np.ndarray:
    K = probs.shape[1]
    cov = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            delta = 1.0 if j == k else 0.0
            cov[j, k] = (scores * probs[:, j] * (delta - probs[:, k])).sum()
    return cov

class MarginalSplitConformal:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_per_class: Optional[np.ndarray] = None

    def fit(self, cal_preds: List[Dict], cal_gt: List[np.ndarray]) -> "MarginalSplitConformal":
        K = cal_gt[0].shape[0]
        scores_per_class = [[] for _ in range(K)]

        for pred, gt in zip(cal_preds, cal_gt):
            s = pred["scores"]
            p = pred["probs"]
            if len(s) == 0:
                
                for k in range(K):
                    scores_per_class[k].append(float(gt[k]))
                continue
            n_pred = pb_count(s, p)
            sigma = np.sqrt(pb_variance(s, p) + 1e-6)
            for k in range(K):
                err = abs(gt[k] - n_pred[k]) / sigma[k]
                scores_per_class[k].append(err)

        self.q_per_class = np.array([
            empirical_quantile(np.array(scores_per_class[k]), self.alpha)
            for k in range(K)
        ])
        return self

    def predict_interval(self, pred: Dict) -> Tuple[np.ndarray, np.ndarray]:
        K = len(self.q_per_class)
        s = pred["scores"]
        p = pred["probs"]
        if len(s) == 0:
            return np.zeros(K), np.zeros(K)
        n_pred = pb_count(s, p)
        sigma = np.sqrt(pb_variance(s, p) + 1e-6)
        lower = np.maximum(0, n_pred - self.q_per_class * sigma)
        upper = n_pred + self.q_per_class * sigma
        return lower, upper

class AdaptiveConformalInference:
    def __init__(self, alpha_target: float = 0.1, gamma: float = 0.05):
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.alpha_t = alpha_target
        self.history_q: List[float] = []
        self.history_scores: List[float] = []  

    def reset(self):
        self.alpha_t = self.alpha_target
        self.history_scores = []

    def update(self, score_t: float, covered_t: bool):
        self.history_scores.append(score_t)
        err_t = 0.0 if covered_t else 1.0
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - err_t)
        
        self.alpha_t = max(1e-3, min(0.5, self.alpha_t))

    def get_quantile(self) -> float:
        if not self.history_scores:
            return 1.0
        return empirical_quantile(np.array(self.history_scores), self.alpha_t)

class ShiftAwareACI(AdaptiveConformalInference):
    def __init__(self, alpha_target: float = 0.1, gamma_0: float = 0.05,
                 lambda_: float = 3.0, gamma_max: float = 0.15):
        super().__init__(alpha_target, gamma_0)
        self.gamma_0 = gamma_0
        self.lambda_ = lambda_
        self.gamma_max = gamma_max
        self.gamma_t_last = gamma_0

    def update(self, score_t: float, covered_t: bool, delta_t: float = 0.0):
        self.history_scores.append(score_t)
        gamma_t = self.gamma_0 * (1.0 + self.lambda_ * max(0.0, delta_t))
        gamma_t = min(gamma_t, self.gamma_max)
        self.gamma_t_last = gamma_t
        err_t = 0.0 if covered_t else 1.0
        self.alpha_t = self.alpha_t + gamma_t * (self.alpha_target - err_t)
        self.alpha_t = max(1e-3, min(0.5, self.alpha_t))

class RollingShiftDetector:
    def __init__(self, window: int = 100, cap: float = 1.0):
        self.window = window
        self.cap = cap
        self.baseline: Optional[float] = None
        self.recent: List[float] = []

    def fit_baseline(self, cal_scores) -> "RollingShiftDetector":
        self.baseline = float(np.median(np.asarray(cal_scores))) + 1e-6
        return self

    def step(self, score_t: float) -> float:
        self.recent.append(float(score_t))
        if len(self.recent) > self.window:
            self.recent.pop(0)
        cur = float(np.median(self.recent))
        delta = (cur - self.baseline) / self.baseline
        return float(np.clip(delta, 0.0, self.cap))

class PBAwareJointConformalOnline:
    def __init__(self, alpha: float = 0.1, window: int = 300):
        self.alpha = alpha
        self.window = window
        self.scores: List[float] = []

    def warmstart(self, cal_scores) -> "PBAwareJointConformalOnline":
        self.scores = list(np.asarray(cal_scores)[-self.window:])
        return self

    def get_quantile(self) -> float:
        if not self.scores:
            return float("inf")
        return empirical_quantile(np.asarray(self.scores[-self.window:]), self.alpha)

    def update(self, score_t: float):
        self.scores.append(float(score_t))
        if len(self.scores) > self.window:
            self.scores = self.scores[-self.window:]

class AdaptivePBJCIOnline:
    def __init__(self, alpha: float = 0.1, w_max: int = 300, w_min: int = 40,
                 m: int = 50, rho_s: float = 0.9, rho_g: float = 1.05,
                 beta: float = 0.03):
        self.alpha = alpha
        self.w_max = w_max
        self.w_min = w_min
        self.m = m
        self.rho_s = rho_s
        self.rho_g = rho_g
        self.beta = beta
        self.scores: List[float] = []
        self.eff = w_max
        self.recent: List[float] = []

    def warmstart(self, cal_scores) -> "AdaptivePBJCIOnline":
        self.scores = list(np.asarray(cal_scores)[-self.w_max:])
        self.eff = min(self.w_max, len(self.scores)) if self.scores else self.w_max
        self.recent = []
        return self

    def get_quantile(self) -> float:
        if not self.scores:
            return float("inf")
        return empirical_quantile(np.asarray(self.scores[-self.eff:]), self.alpha)

    def update(self, score_t: float, covered_t: bool):
        
        self.recent.append(1.0 if covered_t else 0.0)
        if len(self.recent) > self.m:
            self.recent = self.recent[-self.m:]
        rc = float(np.mean(self.recent))
        target = 1.0 - self.alpha
        if rc < target:
            self.eff = max(self.w_min, int(self.eff * self.rho_s))
        elif rc > target + self.beta:
            self.eff = min(self.w_max, int(self.eff * self.rho_g))
        
        self.scores.append(float(score_t))
        if len(self.scores) > self.w_max:
            self.scores = self.scores[-self.w_max:]

def local_coverage_stats(covered_list, window: int = 100) -> Dict[str, float]:
    c = np.asarray(covered_list, dtype=float)
    n = len(c)
    if n == 0:
        return {"min_local_cov": 0.0, "max_miss_run": 0}
    if n >= window:
        roll = np.convolve(c, np.ones(window) / window, mode="valid")
        min_local = float(roll.min())
    else:
        min_local = float(c.mean())
    run = mx = 0
    for v in covered_list:
        run = 0 if v else run + 1
        mx = max(mx, run)
    return {"min_local_cov": min_local, "max_miss_run": int(mx)}

class PBAwareJointConformal:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q: float = 0.0

    def fit(self, cal_preds: List[Dict], cal_gt: List[np.ndarray]) -> "PBAwareJointConformal":
        scores = []
        for pred, gt in zip(cal_preds, cal_gt):
            s = pred["scores"]
            p = pred["probs"]
            K = len(gt)
            if len(s) == 0:
                
                sigma_eps = 1.0
                S_t = max(abs(gt[k]) / sigma_eps for k in range(K))
            else:
                n_pred = pb_count(s, p)
                sigma = np.sqrt(pb_variance(s, p) + 1e-6)
                S_t = max(abs(gt[k] - n_pred[k]) / sigma[k] for k in range(K))
            scores.append(S_t)
        self.q = empirical_quantile(np.array(scores), self.alpha)
        return self

    def predict_interval(self, pred: Dict) -> Tuple[np.ndarray, np.ndarray]:
        s = pred["scores"]
        p = pred["probs"]
        K = pred.get("K", 5)
        if len(s) == 0:
            return np.zeros(K), np.zeros(K)
        n_pred = pb_count(s, p)
        sigma = np.sqrt(pb_variance(s, p) + 1e-6)
        lower = np.maximum(0, n_pred - self.q * sigma)
        upper = n_pred + self.q * sigma
        return lower, upper

class ClassStratifiedConformal:
    def __init__(self, alpha: float = 0.1, bonferroni: bool = True):
        self.alpha = alpha
        self.bonferroni = bonferroni
        self.q_per_class: Optional[np.ndarray] = None

    def fit(self, cal_preds: List[Dict], cal_gt: List[np.ndarray]) -> "ClassStratifiedConformal":
        K = cal_gt[0].shape[0]
        alpha_eff = self.alpha / K if self.bonferroni else self.alpha
        scores_per_class = [[] for _ in range(K)]

        for pred, gt in zip(cal_preds, cal_gt):
            s = pred["scores"]
            p = pred["probs"]
            if len(s) == 0:
                continue
            n_pred = pb_count(s, p)
            sigma = np.sqrt(pb_variance(s, p) + 1e-6)
            for k in range(K):
                if gt[k] > 0:  
                    err = abs(gt[k] - n_pred[k]) / sigma[k]
                    scores_per_class[k].append(err)

        self.q_per_class = np.array([
            empirical_quantile(np.array(scores_per_class[k]) if scores_per_class[k]
                              else np.array([1.0]), alpha_eff)
            for k in range(K)
        ])
        return self

    def predict_interval(self, pred: Dict) -> Tuple[np.ndarray, np.ndarray]:
        K = len(self.q_per_class)
        s = pred["scores"]
        p = pred["probs"]
        if len(s) == 0:
            return np.zeros(K), np.zeros(K)
        n_pred = pb_count(s, p)
        sigma = np.sqrt(pb_variance(s, p) + 1e-6)
        lower = np.maximum(0, n_pred - self.q_per_class * sigma)
        upper = n_pred + self.q_per_class * sigma
        return lower, upper

def coverage_per_class(intervals_lo: np.ndarray, intervals_hi: np.ndarray,
                       gt_counts: np.ndarray) -> np.ndarray:
    covered = (gt_counts >= intervals_lo) & (gt_counts <= intervals_hi)
    return covered.mean(axis=0)

def joint_coverage(intervals_lo: np.ndarray, intervals_hi: np.ndarray,
                   gt_counts: np.ndarray) -> float:
    covered_all = ((gt_counts >= intervals_lo) & (gt_counts <= intervals_hi)).all(axis=1)
    return float(covered_all.mean())

def avg_width_per_class(intervals_lo: np.ndarray, intervals_hi: np.ndarray) -> np.ndarray:
    return (intervals_hi - intervals_lo).mean(axis=0)

def macro_width(intervals_lo: np.ndarray, intervals_hi: np.ndarray) -> float:
    return float(avg_width_per_class(intervals_lo, intervals_hi).mean())

def split_calibration_test(preds: List[Dict], gts: List[np.ndarray],
                           cal_ratio: float = 0.5,
                           seed: int = 42) -> Tuple[List, List, List, List]:
    n = len(preds)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_cal = int(n * cal_ratio)
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]
    cal_preds = [preds[i] for i in cal_idx]
    cal_gt = [gts[i] for i in cal_idx]
    test_preds = [preds[i] for i in test_idx]
    test_gt = [gts[i] for i in test_idx]
    return cal_preds, cal_gt, test_preds, test_gt
