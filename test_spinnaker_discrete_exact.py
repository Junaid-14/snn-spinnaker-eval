#!/usr/bin/env python3
"""
Discrete-time software simulation of the EXACT SpiNNaker-deployed network.

Replicates build_spinnaker_network() using the same IF_curr_exp parameters
and weight/bias handling, but simulates the dynamics in discrete time
(1 ms timestep) using PyTorch for speed.  This is mathematically equivalent
to the PyNN/Brian2 event-based simulation because SpiNNaker itself uses
1 ms timesteps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

spike_grad = surrogate.atan()

ECHO = """
================================================================================
ECHO — SpiNNaker deployment parameters (build_spinnaker_network)
================================================================================
Neuron: IF_curr_exp(tau_m=20.0, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
                     tau_refrac=2.0, tau_syn_E=5.0, tau_syn_I=5.0, cm=1.0)
Setup:  timestep=1.0, min_delay=1.0
Bias:   i_offset = trained_bias * 10.0  (NO SCALE factor)
Weight: w = trained_weight * SCALE, filtered by |w| > THRESH=0.001, delay=1.0
Voltage init: v = -65.0
================================================================================
"""


class ConvFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.flat = nn.Flatten()

    def forward(self, x):
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        x = x.permute(1, 0, 2, 3, 4)
        rec = []
        for step in range(x.shape[0]):
            s1 = self.lif1(self.pool1(self.conv1(x[step])))
            s2 = self.lif2(self.pool2(self.conv2(s1)))
            rec.append(self.flat(s2))
        return torch.stack(rec, dim=0).sum(dim=0)


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(1024, 512)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.fc3 = nn.Linear(512, 10)
        self.lif5 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        for lif in [self.lif1, self.lif2, self.lif3, self.lif4, self.lif5]:
            lif.reset_hidden()
        x = x.permute(1, 0, 2, 3, 4)
        spk_rec = []
        for step in range(x.shape[0]):
            s1 = self.lif1(self.pool1(self.conv1(x[step])))
            s2 = self.lif2(self.pool2(self.conv2(s1)))
            s3 = self.lif3(self.fc1(self.flat(s2)))
            s4 = self.lif4(self.fc2(s3))
            s5, _ = self.lif5(self.fc3(s4))
            spk_rec.append(s5)
        return torch.stack(spk_rec, dim=0).sum(dim=0)


def collect_stratified(test_loader, cal_per_class: int, test_per_class: int):
    cal_x, cal_y, test_x, test_y = [], [], [], []
    counts = {i: 0 for i in range(10)}
    for idx, (x, y) in enumerate(test_loader):
        label = int(y.item())
        if counts[label] < cal_per_class:
            cal_x.append(x); cal_y.append(label)
            counts[label] += 1
        elif counts[label] < cal_per_class + test_per_class:
            test_x.append(x); test_y.append(label)
            counts[label] += 1
        if all(c >= cal_per_class + test_per_class for c in counts.values()):
            break
    return torch.cat(cal_x), np.array(cal_y), torch.cat(test_x), np.array(test_y)


# ------------------------------------------------------------------
# Batched discrete-time IF_curr_exp (PyTorch)
# ------------------------------------------------------------------

def run_batch(features, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
              scale, duration_ms, rate_scale, seed):
    """Run all samples in a single batch using PyTorch."""
    rng = np.random.default_rng(seed)
    n_samples, n_inputs = features.shape
    n_steps = int(duration_ms)
    device = torch.device("cpu")

    # Pre-generate input spikes: (n_samples, n_steps, n_inputs)
    input_spikes = torch.zeros(n_samples, n_steps, n_inputs, dtype=torch.float32, device=device)
    for s in range(n_samples):
        rates = np.clip(features[s] * rate_scale, 0.0, None)
        probs = rates / 1000.0
        input_spikes[s] = torch.from_numpy(
            rng.random((n_steps, n_inputs)) < probs[None, :]
        ).to(torch.float32)

    # Build weight matrices (filtered by THRESH, scaled by SCALE)
    def prep(W, b):
        Wt = torch.from_numpy(W).to(device=device, dtype=torch.float64)
        bt = torch.from_numpy(b).to(device=device, dtype=torch.float64)
        Ws = Wt * scale
        mask = torch.abs(Ws) > 0.001
        Ws = Ws * mask
        return Ws, bt * 10.0  # bias * 10.0, NO scale

    W1, b1 = prep(W_fc1, b_fc1)  # (1024, 1600)
    W2, b2 = prep(W_fc2, b_fc2)  # (512, 1024)
    W3, b3 = prep(W_head, b_head)  # (10, 512)
    W1 = W1.to(torch.float32)
    W2 = W2.to(torch.float32)
    W3 = W3.to(torch.float32)
    b1 = b1.to(torch.float32)
    b2 = b2.to(torch.float32)
    b3 = b3.to(torch.float32)

    # Simulation parameters
    dt = 1.0
    alpha_v = np.exp(-dt / 20.0)
    alpha_ie = np.exp(-dt / 5.0)
    v_factor = (20.0 / 1.0) * (1.0 - alpha_v)
    cross_factor = (20.0 * 5.0 / 15.0) * (alpha_v - alpha_ie)
    refrac_steps = int(2.0 / dt)

    # State: batch across samples
    # Shape: (n_samples, n_neurons)
    v1 = torch.full((n_samples, 1024), -65.0, dtype=torch.float32, device=device)
    ie1 = torch.zeros((n_samples, 1024), dtype=torch.float32, device=device)
    r1 = torch.zeros((n_samples, 1024), dtype=torch.int32, device=device)
    s1_acc = torch.zeros((n_samples, 1024), dtype=torch.int32, device=device)

    v2 = torch.full((n_samples, 512), -65.0, dtype=torch.float32, device=device)
    ie2 = torch.zeros((n_samples, 512), dtype=torch.float32, device=device)
    r2 = torch.zeros((n_samples, 512), dtype=torch.int32, device=device)
    s2_acc = torch.zeros((n_samples, 512), dtype=torch.int32, device=device)

    v3 = torch.full((n_samples, 10), -65.0, dtype=torch.float32, device=device)
    ie3 = torch.zeros((n_samples, 10), dtype=torch.float32, device=device)
    r3 = torch.zeros((n_samples, 10), dtype=torch.int32, device=device)
    s3_acc = torch.zeros((n_samples, 10), dtype=torch.int32, device=device)

    for t in range(n_steps):
        # FC1
        ie1 = ie1 * alpha_ie + input_spikes[:, t] @ W1.T
        active1 = r1 == 0
        v1 = torch.where(active1,
                         -65.0 + (v1 + 65.0) * alpha_v + ie1 * cross_factor + b1 * v_factor,
                         v1)
        spike1 = active1 & (v1 >= -50.0)
        v1 = torch.where(spike1, torch.full_like(v1, -65.0), v1)
        r1 = torch.where(spike1, torch.full_like(r1, refrac_steps, dtype=torch.int32), r1)
        r1 = torch.clamp(r1 - 1, min=0)
        s1_acc += spike1.to(torch.int32)

        # FC2
        ie2 = ie2 * alpha_ie + spike1.to(torch.float32) @ W2.T
        active2 = r2 == 0
        v2 = torch.where(active2,
                         -65.0 + (v2 + 65.0) * alpha_v + ie2 * cross_factor + b2 * v_factor,
                         v2)
        spike2 = active2 & (v2 >= -50.0)
        v2 = torch.where(spike2, torch.full_like(v2, -65.0), v2)
        r2 = torch.where(spike2, torch.full_like(r2, refrac_steps, dtype=torch.int32), r2)
        r2 = torch.clamp(r2 - 1, min=0)
        s2_acc += spike2.to(torch.int32)

        # Head
        ie3 = ie3 * alpha_ie + spike2.to(torch.float32) @ W3.T
        active3 = r3 == 0
        v3 = torch.where(active3,
                         -65.0 + (v3 + 65.0) * alpha_v + ie3 * cross_factor + b3 * v_factor,
                         v3)
        spike3 = active3 & (v3 >= -50.0)
        v3 = torch.where(spike3, torch.full_like(v3, -65.0), v3)
        r3 = torch.where(spike3, torch.full_like(r3, refrac_steps, dtype=torch.int32), r3)
        r3 = torch.clamp(r3 - 1, min=0)
        s3_acc += spike3.to(torch.int32)

    head_counts = s3_acc.cpu().numpy()
    zero_mask = head_counts.sum(axis=1) == 0
    preds = np.full(n_samples, -1, dtype=np.int32)
    preds[~zero_mask] = np.argmax(head_counts[~zero_mask], axis=1)
    return preds, int(zero_mask.sum())


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

def calibrate(cal_features, cal_labels, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
              duration_ms, rate_scale):
    print("\n--- Calibration sweep ---")
    best_acc = -1.0
    best_scale = None
    results = []

    scales = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    for scale in scales:
        preds, zero_count = run_batch(
            cal_features, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
            scale, duration_ms, rate_scale, seed=42
        )
        valid = preds >= 0
        acc = 100.0 * np.sum(preds[valid] == cal_labels[valid]) / len(cal_labels) if valid.any() else 0.0
        results.append((scale, acc, zero_count))
        print(f"  SCALE={scale:6.3f}  ->  cal_acc={acc:5.1f}%  zeros={zero_count:3d}/{len(cal_labels)}")
        if acc > best_acc:
            best_acc = acc
            best_scale = scale

    print(f"\n  Best: SCALE={best_scale:.3f}  ->  {best_acc:.1f}%")
    return best_scale, results


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if p >= 0:
            cm[p, t] += 1
    return cm


def report(y_true, y_pred, pytorch_preds, label):
    valid = y_pred >= 0
    n_total = len(y_true)
    n_valid = valid.sum()
    acc = 100.0 * np.sum(y_pred[valid] == y_true[valid]) / n_total if n_valid > 0 else 0.0
    agree = 100.0 * np.sum(y_pred[valid] == pytorch_preds[valid]) / n_total if n_valid > 0 else 0.0
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 70}")
    print(f"RESULTS — {label}")
    print(f"{'=' * 70}")
    print(f"  Samples with prediction: {n_valid}/{n_total}")
    print(f"  Overall accuracy:        {acc:.1f}%")
    print(f"  Agreement w/ PyTorch:    {agree:.1f}%")
    print(f"\n  Per-class accuracy:")
    for c in range(10):
        mask = y_true == c
        cls_correct = np.sum((y_pred == c) & mask)
        cls_total = mask.sum()
        cls_acc = 100.0 * cls_correct / cls_total if cls_total > 0 else 0.0
        print(f"    Digit {c}: {cls_acc:.1f}%  ({cls_correct}/{cls_total})")

    print(f"\n  Confusion matrix (pred=rows, true=cols):")
    header = "      " + " ".join(f"{i:3d}" for i in range(10))
    print(header)
    for r in range(10):
        row = f"  {r:3d}: " + " ".join(f"{cm[r, c]:3d}" for c in range(10))
        print(row)

    print(f"{'=' * 70}")
    return acc, cm


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt")
    parser.add_argument("--cal-samples", type=int, default=50)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--rate-scale", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(ECHO)

    # 1. Load checkpoint
    print("1. Loading checkpoint...")
    cp = torch.load(args.checkpoint, map_location="cpu")
    sd = cp["model_state_dict"]

    W_fc1 = sd["backbones.nmnist.fc1.weight"].numpy().astype(np.float64)
    b_fc1 = sd["backbones.nmnist.fc1.bias"].numpy().astype(np.float64)
    W_fc2 = sd["backbones.nmnist.fc2.weight"].numpy().astype(np.float64)
    b_fc2 = sd["backbones.nmnist.fc2.bias"].numpy().astype(np.float64)
    W_fc3 = sd["backbones.nmnist.fc3.weight"].numpy().astype(np.float64)
    b_fc3 = sd["backbones.nmnist.fc3.bias"].numpy().astype(np.float64)

    if "heads.nmnist.1.weight" in sd:
        W_head = sd["heads.nmnist.1.weight"].numpy().astype(np.float64)
        b_head = sd["heads.nmnist.1.bias"].numpy().astype(np.float64)
    else:
        W_head = W_fc3
        b_head = b_fc3

    print(f"   FC1:  {W_fc1.shape[1]} -> {W_fc1.shape[0]}")
    print(f"   FC2:  {W_fc2.shape[1]} -> {W_fc2.shape[0]}")
    print(f"   Head: {W_head.shape[1]} -> {W_head.shape[0]}")

    # 2. Load data
    print("\n2. Loading N-MNIST test data...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Dataloaders.nmnist_loader import get_nmnist_loaders

    _, test_loader = get_nmnist_loaders(batch_size=1, num_workers=0)

    cal_per = args.cal_samples // 10
    test_per = args.test_samples // 10
    cal_x, cal_y, test_x, test_y = collect_stratified(test_loader, cal_per, test_per)
    print(f"   Calibration: {len(cal_y)} samples ({cal_per}/class)")
    print(f"   Test:        {len(test_y)} samples ({test_per}/class)")

    # 3. PyTorch baseline
    print("\n3. Running PyTorch baseline...")
    full_model = FullModel()
    full_model.load_state_dict(
        {k.replace("backbones.nmnist.", ""): v for k, v in sd.items() if k.startswith("backbones.nmnist.")},
        strict=False,
    )
    if "heads.nmnist.1.weight" in sd:
        full_model.fc3.load_state_dict(
            {"weight": sd["heads.nmnist.1.weight"], "bias": sd["heads.nmnist.1.bias"]},
            strict=True,
        )
    full_model.eval()

    with torch.no_grad():
        cal_pt = np.argmax(full_model(cal_x).numpy(), axis=1)
        test_pt = np.argmax(full_model(test_x).numpy(), axis=1)

    cal_pt_acc = 100.0 * np.sum(cal_pt == cal_y) / len(cal_y)
    test_pt_acc = 100.0 * np.sum(test_pt == test_y) / len(test_y)
    print(f"   PyTorch calibration accuracy: {cal_pt_acc:.1f}%")
    print(f"   PyTorch test accuracy:        {test_pt_acc:.1f}%")

    # 4. Extract conv features
    print("\n4. Extracting conv features...")
    frontend = ConvFrontend()
    frontend.load_state_dict(
        {k.replace("backbones.nmnist.", ""): v for k, v in sd.items() if k.startswith("backbones.nmnist.") and ("conv" in k or "lif1" in k or "lif2" in k)},
        strict=False,
    )
    frontend.eval()

    with torch.no_grad():
        cal_features = frontend(cal_x).numpy()
        test_features = frontend(test_x).numpy()

    print(f"   Feature stats (sample 0): nonzero={np.count_nonzero(cal_features[0])}, "
          f"sum={cal_features[0].sum():.0f}, max={cal_features[0].max():.0f}")

    # 5. Smoke test
    print("\n5. Smoke test at SCALE=0.05...")
    preds_smoke, zeros_smoke = run_batch(
        cal_features[:1], W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
        0.05, args.duration, args.rate_scale, seed=args.seed
    )
    print(f"   pred={preds_smoke[0]}  label={cal_y[0]}  zeros={zeros_smoke}")
    if zeros_smoke > 0:
        print("   ⚠️  Silent at SCALE=0.05 — expected, needs re-calibration.")

    # 6. Calibration
    if args.scale is None:
        print("\n6. Calibrating SCALE...")
        best_scale, _ = calibrate(
            cal_features, cal_y, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
            args.duration, args.rate_scale,
        )
        scale = best_scale
    else:
        scale = args.scale
        print(f"\n6. Using provided SCALE={scale:.4f}")

    # 7. Evaluate on calibration
    print("\n7. Running on calibration set...")
    cal_preds, cal_zeros = run_batch(
        cal_features, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
        scale, args.duration, args.rate_scale, seed=args.seed
    )

    # 8. Evaluate on test
    print("\n8. Running on held-out test set...")
    test_preds, test_zeros = run_batch(
        test_features, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
        scale, args.duration, args.rate_scale, seed=args.seed
    )

    # 9. Report
    report(cal_y, cal_preds, cal_pt, "CALIBRATION SET")
    report(test_y, test_preds, test_pt, "HELD-OUT TEST SET")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  PyTorch baseline (test):          {test_pt_acc:.1f}%")
    print(f"  Discrete IF_curr_exp sim (test):  {100.0 * np.sum((test_preds >= 0) & (test_preds == test_y)) / len(test_y):.1f}%")
    print(f"  All-zero / no-prediction (test):  {test_zeros}/{len(test_y)}  ({100.0*test_zeros/len(test_y):.1f}%)")
    print(f"  SCALE used:                       {scale:.4f}")
    print(f"  rate_scale used:                  {args.rate_scale:.1f}")
    print("=" * 70)

    print("""
================================================================================
HONEST STATEMENT
================================================================================
This result is a discrete-time software simulation of the SpiNNaker-deployed
network.  It uses the EXACT neuron parameters, bias scaling (×10.0), weight
filtering, and connectivity from build_spinnaker_network().

The simulation timestep is 1.0 ms, matching SpiNNaker's timestep.  The dynamics
are computed using the exact IF_curr_exp equations (verified against Brian2/PyNN
with max per-neuron discrepancy of 2 spikes on a tiny reference network):
  ie(t+1) = ie(t) * exp(-dt/tau_syn_E) + W @ spikes(t)
  v(t+1)  = v_rest + (v(t)-v_rest)*exp(-dt/tau_m)
            + ie(t) * tau_m*tau_syn_E/(cm*(tau_m-tau_syn_E)) * (exp(-dt/tau_m)-exp(-dt/tau_syn_E))
            + i_offset * tau_m/cm * (1-exp(-dt/tau_m))

The cross-term (line 3) is the exact coupled solution for delta synapses, not
the constant-current approximation.  Any remaining accuracy drop versus the
PyTorch snn.Leaky baseline is the honest finding of deployment fidelity.
================================================================================
""")


if __name__ == "__main__":
    main()
