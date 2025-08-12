"""
Offline Bradley-Terry (Elo) utilities.

Only imported by analysis scripts; the live debate engine never touches
this file, so it has no impact on token usage or runtime cost.
"""

import math
import json
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

_WINNER_LINE = re.compile(r'^\s*WINNER:\s*(.+?)\s*$', re.I | re.M)


# ---------- helper: build win-matrix from a log ------------------------

def _winner_label(match):
    # Preferred: structured field written by judge_module
    w = match.get("winner") or match.get("judge_evaluation", {}).get("winner")
    if w:
        return w
    # Fallback: parse strict WINNER line from verdict text
    v = match.get("verdict") or match.get("judge_evaluation", {}).get("verdict", "")
    m = _WINNER_LINE.search(v)
    return m.group(1).strip() if m else None


def win_matrix_from_log(log_path: str, debater_ids):
    """
    Build a winner matrix W where W[i,j] = wins of debater debater_ids[i] over debater_ids[j].
    Robust to neutral labels ("Strategy 1/2") and to older logs.
    """
    with open(log_path, "r") as f:
        log = json.load(f)

    id2idx = {d: i for i, d in enumerate(debater_ids)}
    W = np.zeros((len(debater_ids), len(debater_ids)), dtype=int)

    for m in log.get("matches", []):
        wlab = _winner_label(m)
        if not wlab:
            continue  # skip if no clear winner

        # Preferred mapping: explicit neutral mapping (if present in newer logs)
        side2id = m.get("side_to_debater_id")
        if not side2id:
            # Fallback mapping from first two speakers to legacy fields
            turns = m.get("turns", [])
            if len(turns) < 2:
                continue
            s1 = turns[0]["speaker"]
            s2 = turns[1]["speaker"]
            pro = m.get("debater_pro")
            con = m.get("debater_con")
            if pro is None or con is None:
                continue
            side2id = {s1: pro, s2: con}

        if wlab not in side2id:
            continue  # label mismatch; skip defensively

        win_id = side2id[wlab]
        # loser is whichever side isn't the winner
        lose_ids = [v for k, v in side2id.items() if k != wlab]
        if not lose_ids:
            continue
        lose_id = lose_ids[0]

        if win_id in id2idx and lose_id in id2idx:
            W[id2idx[win_id], id2idx[lose_id]] += 1

    # --- DEBUG: inspect win-count matrix ---------------------------
    print("\nWin matrix (rows = winners, cols = losers)\n", W, "\n")
    # ---------------------------------------------------------------
    return W


# ---------- Bradley–Terry negative log-likelihood + gradient -----------

def _bt_nll(E: np.ndarray, w: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Negative log-likelihood and gradient for the Bradley-Terry model.
    E shape (n,), w shape (n,n).
    """
    n = len(E)
    nll = 0.0
    grad = np.zeros_like(E)

    for i in range(n):
        for j in range(i + 1, n):
            if w[i, j] + w[j, i] == 0:
                continue

            d_raw = E[i] - E[j]
            d = np.clip(d_raw, -20.0, 20.0)

            p      = expit(d)                    # σ(d)
            logp   = -np.logaddexp(0, -d)        # log σ(d)
            log1p  = -np.logaddexp(0,  d)        # log σ(-d)

            nll  -= w[i, j] * logp + w[j, i] * log1p
            grad[i] -= w[i, j] * (1 - p) - w[j, i] * p
            grad[j] += w[i, j] * (1 - p) - w[j, i] * p

    return nll, grad


# ---------- public fitter ---------------------------------------------

def fit_bt(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Offline Bradley–Terry fit by BFGS with the identifiability constraint
        Σ_i E_i = 0
    handled explicitly.

    • We optimise over x ∈ ℝ^{n-1};  the nth rating is the negative sum
      so the full vector lies in the (n-1)-dimensional zero-mean subspace.
    • Returns (E_hat, covariance), where the covariance is the inverse
      Hessian lifted back to the full n×n space.
    """
    n = w.shape[0]
    g0 = _bt_nll(np.zeros(n), w)[1]
    print("grad@0  =", np.round(g0, 3))

    # -------- helper: expand reduced coords to full length -------------
    def unpack(x_red: np.ndarray) -> np.ndarray:
        # x_red = [E₁ … E_{n-1}],  enforce E_n = –Σ_{i<n} E_i
        e_last = -float(np.sum(x_red))
        return np.concatenate([x_red, [e_last]])

    # -------- objective & gradient in reduced coordinates --------------
    def objective(x_red: np.ndarray):
        E = unpack(x_red)
        nll, g_full = _bt_nll(E, w)     # g_full ∈ ℝⁿ

        # Gradient wrt x_red is g_full for first n-1 coords,
        # minus g_full[n] because dE_n/dx_k = –1
        grad_red = g_full[: n - 1] - g_full[-1]
        return nll, grad_red

    # -------- run BFGS --------------------------------------------------
    res = minimize(lambda x: objective(x)[0],
                   x0=np.zeros(n - 1),
                   jac=lambda x: objective(x)[1],
                   method="BFGS")
    
    print("BFGS success:", res.success, res.message)
    print("Finished x  :", np.round(res.x, 6))

    E_hat = unpack(res.x)

    # -------- lift (n-1)×(n-1) Hess-inv to full n×n covariance ---------
    H_inv = res.hess_inv          # shape (n-1, n-1)
    cov = np.zeros((n, n))
    cov[: n - 1, : n - 1] = H_inv
    cov[:-1, -1] = cov[-1, :-1] = -H_inv.sum(axis=1)
    cov[-1, -1] =  H_inv.sum()

    return E_hat, cov