"""Pooled multi-task neural network (MLP) with three ways of using combustion stoichiometry.

variant U : outputs CO2 log-growth (5) and fuel-energy log-growth (3); plain MSE.
variant S : U + soft penalty  lam * mean_coupled (log C_hat - log E_hat - log EF0)^2
variant H : outputs energy log-growth (3) and direct CO2 heads (5); for coupled fuels the CO2
            forecast is EF0 * E_hat inside the network (hard, by construction).
The post-hoc variant (P) is U followed by projection, done in assemble.py.

Training: Adam, early stopping on the last three target years before the origin, then a refit
on all rows for the selected number of epochs.
"""
import argparse
import pickle
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")
from features import C_OFF, E_OFF, coupled_mask  # noqa: E402

torch.set_num_threads(2)


class MLP(nn.Module):
    def __init__(self, d_in, width, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(width, 8)

    def forward(self, x):
        return self.head(self.body(x))


def outputs(model, x, variant, c0, e0, cm):
    """Return (gc (n,5), ge (n,3)) growth predictions as torch tensors."""
    o = model(x)
    if variant in ("U", "S"):
        return o[:, :5], o[:, 5:]
    ge = o[:, :3]
    gc_dir = o[:, 3:]
    e_hat = (e0 + E_OFF) * torch.exp(ge) - E_OFF
    ef0 = torch.where(cm, c0[:, :3] / torch.clamp(e0, min=1e-9), torch.zeros_like(e0))
    c_struct = torch.clamp(ef0 * e_hat, min=0.0)
    gc_struct = torch.log(c_struct + C_OFF) - torch.log(c0[:, :3] + C_OFF)
    gc_fuel = torch.where(cm, gc_struct, gc_dir[:, :3])
    return torch.cat([gc_fuel, gc_dir[:, 3:]], dim=1), ge


def loss_fn(gc, ge, yc, ye, variant, lam, c0, e0, cm):
    mc = torch.isfinite(yc)
    me = torch.isfinite(ye)
    lc = ((gc - torch.nan_to_num(yc)) ** 2)[mc].mean()
    le = ((ge - torch.nan_to_num(ye)) ** 2)[me].mean()
    loss = lc + le
    if variant == "S" and lam > 0:
        log_c = torch.log(torch.clamp((c0[:, :3] + C_OFF) * torch.exp(gc[:, :3]) - C_OFF, min=1e-6))
        log_e = torch.log(torch.clamp((e0 + E_OFF) * torch.exp(ge) - E_OFF, min=1e-6))
        log_ef0 = torch.log(torch.clamp(c0[:, :3] / torch.clamp(e0, min=1e-9), min=1e-9))
        v = (log_c - log_e - log_ef0) ** 2
        loss = loss + lam * v[cm].mean()
    return loss


def train(Xtr, yc, ye, c0, e0, variant, width, dropout, lam, seed, epochs, bs=256, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(Xtr.shape[1], width, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr)
    tens = [torch.tensor(a, dtype=torch.float32) for a in (Xtr, yc, ye, c0, e0)]
    cm = torch.tensor(coupled_mask(c0, e0))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            gc, ge = outputs(model, tens[0][b], variant, tens[3][b], tens[4][b], cm[b])
            loss = loss_fn(gc, ge, tens[1][b], tens[2][b], variant, lam, tens[3][b], tens[4][b], cm[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def evaluate_loss(model, X, yc, ye, c0, e0, variant):
    model.eval()
    with torch.no_grad():
        t = [torch.tensor(a, dtype=torch.float32) for a in (X, yc, ye, c0, e0)]
        cm = torch.tensor(coupled_mask(c0, e0))
        gc, ge = outputs(model, t[0], variant, t[3], t[4], cm)
        return loss_fn(gc, ge, t[1], t[2], "U", 0.0, t[3], t[4], cm).item()   # accuracy only


def select_epochs(Xtr, yc, ye, c0, e0, va, variant, width, dropout, lam, seed, max_ep=200, patience=20, bs=256):
    """Early stopping on a temporal validation block; returns the best epoch count."""
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], width, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr = ~va
    tens = [torch.tensor(a[tr], dtype=torch.float32) for a in (Xtr, yc, ye, c0, e0)]
    cm = torch.tensor(coupled_mask(c0[tr], e0[tr]))
    n = int(tr.sum())
    best, best_ep, bad = np.inf, 1, 0
    for ep in range(1, max_ep + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            gc, ge = outputs(model, tens[0][b], variant, tens[3][b], tens[4][b], cm[b])
            loss = loss_fn(gc, ge, tens[1][b], tens[2][b], variant, lam, tens[3][b], tens[4][b], cm[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = evaluate_loss(model, Xtr[va], yc[va], ye[va], c0[va], e0[va], variant)
        if v < best - 1e-6:
            best, best_ep, bad = v, ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_ep


def run_origin(meta, X, T, T0, variant, width, dropout, lam, seed):
    tr = np.where(meta.target_year.values <= T0)[0]
    te = np.where(meta.tau.values == T0)[0]
    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    va = meta.target_year.values[tr] > T0 - 3
    args = (Xs[tr], T["g_c"][tr], T["g_e"][tr], T["c0"][tr], T["e0"][tr])
    ep = select_epochs(*args, va, variant, width, dropout, lam, seed)
    model = train(*args, variant, width, dropout, lam, seed, epochs=ep)
    model.eval()
    with torch.no_grad():
        c0 = torch.tensor(T["c0"][te], dtype=torch.float32)
        e0 = torch.tensor(T["e0"][te], dtype=torch.float32)
        cm = torch.tensor(coupled_mask(T["c0"][te], T["e0"][te]))
        gc, ge = outputs(model, torch.tensor(Xs[te], dtype=torch.float32), variant, c0, e0, cm)
    return dict(idx=te, gc=gc.numpy(), ge=ge.numpy(), epochs=ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["U", "S", "H"])
    ap.add_argument("--origins", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.origins.split("-"))
    seeds = [int(s) for s in a.seeds.split(",")]
    meta, X, T = pickle.load(open("results/samples.pkl", "rb"))
    res = {}
    t0 = time.time()
    for T0 in range(lo, hi + 1):
        for s in seeds:
            res[(T0, s)] = run_origin(meta, X, T, T0, a.variant, a.width, a.dropout, a.lam, s)
        print(f"{a.variant} origin {T0} done ({time.time() - t0:.0f}s, epochs {res[(T0, seeds[-1])]['epochs']})", flush=True)
    pickle.dump(dict(cfg=vars(a), res=res), open(a.out, "wb"))


if __name__ == "__main__":
    main()
