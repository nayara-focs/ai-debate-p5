"""
Offline Bradley-Terry (Elo) utilities.

Only imported by analysis scripts; the live debate engine never touches
this file, so it has no impact on token usage or runtime cost.
"""

import math
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


# ---------- helper: build win-matrix from a log ------------------------

def win_matrix_from_log(log_path: str, debater_ids: List[str]) -> np.ndarray:
    """
    Parse the JSON log saved by run_debate.py and return a matrix
    w[i,j] = number of times debater i beat debater j.
    Requires that each match dict stores:
        "debater_pro", "debater_con", "verdict".
    """

    with Path(log_path).open("r") as f:
        data = json.load(f)

    # The run_debate script stores {"matches": [...]}.
    # Accept either wrapped or bare list for robustness.
    matches = data["matches"] if isinstance(data, dict) and "matches" in data else data

    id2idx = {d: k for k, d in enumerate(debater_ids)}
    w = np.zeros((len(debater_ids), len(debater_ids)), dtype=int)

    for m in matches:
        pro = m["debater_pro"]
        con = m["debater_con"]
        verdict = m["verdict"]

        v_lc = verdict.lower()

        if "pro-p5 wins" in v_lc:
            winner, loser = pro, con
        elif "against-p5 wins" in v_lc:
            winner, loser = con, pro
        else:
            # unrecognised verdict → skip
            continue
        
        w[id2idx[winner], id2idx[loser]] += 1
    
    # --- DEBUG: inspect win-count matrix ---------------------------
    print("\nWin matrix (rows = winners, cols = losers)\n", w, "\n")
    # ---------------------------------------------------------------
    return w


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