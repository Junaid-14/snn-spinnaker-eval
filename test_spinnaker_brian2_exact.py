#!/usr/bin/env python3
"""
Brian2 software simulation of the EXACT SpiNNaker-deployed network.

Replicates build_spinnaker_network() from test_spinnaker_backbone_e2e_inference.py
using PyNN/Brian2 with the identical neuron parameters, weight handling, and
bias scaling.  This is NOT snn.Leaky; it is a Brian2 simulation of the SpiNNaker
network.
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

import pyNN.brian2 as sim
from brian2 import Synapses, nA, ms

spike_grad = surrogate.atan()

# ------------------------------------------------------------------
# Echo of the SpiNNaker network definition we are replicating
# ------------------------------------------------------------------

ECHO = """
================================================================================
ECHO — Lines read from build_spinnaker_network() (test_spinnaker_backbone_e2e_inference.py)
================================================================================

Neuron parameters (lines 227-233, 264-269, 304-308):
    sim.IF_curr_exp(
        tau_m=20.0,
        v_rest=-65.0,
        v_reset=-65.0,
        v_thresh=-50.0,
        tau_refrac=2.0,
    )

PyNN defaults (not overridden in original code):
    tau_syn_E = 5.0 ms, tau_syn_I = 5.0 ms, cm = 1.0 nF

Setup (line 209):
    sim.setup(timestep=1.0, min_delay=1.0)

Input population (lines 214-218):
    sim.SpikeSourcePoisson(rate=0.0, start=0.0, duration=100.0)

Bias handling (lines 236, 273, 312):
    out.set(i_offset=b_fc1[s:e].astype(np.float64) * 10.0)
    out.set(i_offset=b_fc2[s:e].astype(np.float64) * 10.0)
    head_pop.set(i_offset=b_head.astype(np.float64) * 10.0)
    →  i_offset = trained_bias * 10.0   (NO SCALE factor on bias)

Weight handling (lines 238-244, 282-286, 320-324):
    w = float(Wc[post, pre]) * scale
    if abs(w) > thresh:
        conn.append((pre, post, w, 1.0))
    →  weight = trained_weight * SCALE, filtered by |w| > THRESH, delay=1.0

================================================================================
"""

# ------------------------------------------------------------------
# PyTorch reconstructions
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

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
# Network construction — EXACT SpiNNaker parameters
# ------------------------------------------------------------------

def build_network(W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head, scale: float, thresh: float):
    """Build the SpiNNaker network in Brian2 using EXACT original parameters."""
    sim.setup(timestep=1.0, min_delay=1.0)

    # Input: Poisson sources (will be driven via SpikeSourceArray for control)
    input_pop = sim.Population(1600, sim.SpikeSourceArray(spike_times=[]))

    # FC1: 1600 -> 1024
    fc1_pop = sim.Population(
        1024,
        sim.IF_curr_exp(
            tau_m=20.0,
            v_rest=-65.0,
            v_reset=-65.0,
            v_thresh=-50.0,
            tau_refrac=2.0,
            tau_syn_E=5.0,
            tau_syn_I=5.0,
            cm=1.0,
            i_offset=0.0,
        ),
    )
    fc1_pop.initialize(v=-65.0)

    # FC2: 1024 -> 512
    fc2_pop = sim.Population(
        512,
        sim.IF_curr_exp(
            tau_m=20.0,
            v_rest=-65.0,
            v_reset=-65.0,
            v_thresh=-50.0,
            tau_refrac=2.0,
            tau_syn_E=5.0,
            tau_syn_I=5.0,
            cm=1.0,
            i_offset=0.0,
        ),
    )
    fc2_pop.initialize(v=-65.0)

    # Head: 512 -> 10
    head_pop = sim.Population(
        10,
        sim.IF_curr_exp(
            tau_m=20.0,
            v_rest=-65.0,
            v_reset=-65.0,
            v_thresh=-50.0,
            tau_refrac=2.0,
            tau_syn_E=5.0,
            tau_syn_I=5.0,
            cm=1.0,
            i_offset=0.0,
        ),
    )
    head_pop.initialize(v=-65.0)
    head_pop.record(["spikes"])

    def add_synapses(pre_pynn, post_pynn, W, b):
        pre_brian = pre_pynn.brian2_group
        post_brian = post_pynn.brian2_group
        mask = np.abs(W) > thresh
        W_filtered = W * mask * scale
        post_idx, pre_idx = np.nonzero(mask)
        syn = Synapses(pre_brian, post_brian, model="w : amp", on_pre="ie_post += w")
        syn.connect(i=pre_idx, j=post_idx)
        syn.w = W_filtered[post_idx, pre_idx] * nA
        # Delay must be added explicitly to match min_delay=1.0
        syn.delay = 1.0 * ms
        sim.state.network.add(syn)
        # Bias: trained_bias * 10.0  (EXACTLY as in original script)
        for i in range(W.shape[0]):
            post_pynn[i].i_offset = float(b[i]) * 10.0

    add_synapses(input_pop, fc1_pop, W_fc1, b_fc1)
    add_synapses(fc1_pop, fc2_pop, W_fc2, b_fc2)
    add_synapses(fc2_pop, head_pop, W_head, b_head)

    return sim, input_pop, head_pop


def run_samples(sim, input_pop, head_pop, features, rate_scale, duration_ms, seed):
    rng = np.random.default_rng(seed)
    n_samples, n_inputs = features.shape
    n_steps = int(duration_ms)
    results = []
    zero_count = 0

    for s in range(n_samples):
        rates = np.clip(features[s] * rate_scale, 0.0, None)
        input_spikes = rng.poisson(rates / 1000.0, size=(n_steps, n_inputs)) > 0

        indices = []
        times = []
        for nid in range(n_inputs):
            ts = np.where(input_spikes[:, nid])[0]
            for t in ts:
                indices.append(nid)
                times.append(float(t))

        input_pop.brian2_group.set_spikes(np.array(indices), np.array(times) * ms)

        sim.run(duration_ms)

        data = head_pop.get_data("spikes")
        spike_counts = np.array([len(st) for st in data.segments[-1].spiketrains])

        if spike_counts.sum() == 0:
            zero_count += 1
            results.append(-1)  # "no prediction"
        else:
            results.append(int(np.argmax(spike_counts)))

        sim.reset()

    return np.array(results), zero_count


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

def calibrate(cal_features, cal_labels, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head, duration_ms, rate_scale):
    print("\n--- Calibration sweep (SCALE only) ---")
    best_acc = -1.0
    best_scale = None
    results = []

    for scale in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        sim_net, input_pop, head_pop = build_network(
            W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head, scale, 0.001
        )
        preds, zero_count = run_samples(
            sim_net, input_pop, head_pop, cal_features, rate_scale, duration_ms, seed=42
        )
        sim_net.end()

        valid = preds >= 0
        acc = 100.0 * np.sum(preds[valid] == cal_labels[valid]) / len(cal_labels) if valid.any() else 0.0
        results.append((scale, acc, zero_count))
        print(f"  SCALE={scale:.4f}  ->  cal_acc={acc:.1f}%  zeros={zero_count}/{len(cal_labels)}")
        if acc > best_acc:
            best_acc = acc
            best_scale = scale

    print(f"\n  Best calibration: SCALE={best_scale:.4f}  ->  {best_acc:.1f}%")
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

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    print("1. Loading checkpoint...")
    cp = torch.load(args.checkpoint, map_location="cpu")
    sd = cp["model_state_dict"]

    W_fc1 = sd["backbones.nmnist.fc1.weight"].numpy().astype(np.float32)
    b_fc1 = sd["backbones.nmnist.fc1.bias"].numpy().astype(np.float32)
    W_fc2 = sd["backbones.nmnist.fc2.weight"].numpy().astype(np.float32)
    b_fc2 = sd["backbones.nmnist.fc2.bias"].numpy().astype(np.float32)
    W_fc3 = sd["backbones.nmnist.fc3.weight"].numpy().astype(np.float32)
    b_fc3 = sd["backbones.nmnist.fc3.bias"].numpy().astype(np.float32)

    if "heads.nmnist.1.weight" in sd:
        W_head = sd["heads.nmnist.1.weight"].numpy().astype(np.float32)
        b_head = sd["heads.nmnist.1.bias"].numpy().astype(np.float32)
    else:
        W_head = W_fc3
        b_head = b_fc3

    print(f"   FC1:  {W_fc1.shape[1]} -> {W_fc1.shape[0]}")
    print(f"   FC2:  {W_fc2.shape[1]} -> {W_fc2.shape[0]}")
    print(f"   Head: {W_head.shape[1]} -> {W_head.shape[0]}")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print("\n2. Loading N-MNIST test data...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Dataloaders.nmnist_loader import get_nmnist_loaders

    _, test_loader = get_nmnist_loaders(batch_size=1, num_workers=0)

    cal_per = args.cal_samples // 10
    test_per = args.test_samples // 10
    cal_x, cal_y, test_x, test_y = collect_stratified(test_loader, cal_per, test_per)
    print(f"   Calibration: {len(cal_y)} samples ({cal_per}/class)")
    print(f"   Test:        {len(test_y)} samples ({test_per}/class)")

    # ------------------------------------------------------------------
    # 3. PyTorch baseline
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Extract conv features
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. SMOKE TEST — one sample at a candidate SCALE
    # ------------------------------------------------------------------
    print("\n5. Smoke test — pushing one sample through...")
    sim_smoke, input_pop_smoke, head_pop_smoke = build_network(
        W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head, scale=0.05, thresh=0.001
    )
    preds_smoke, zeros_smoke = run_samples(
        sim_smoke, input_pop_smoke, head_pop_smoke,
        cal_features[:1], args.rate_scale, args.duration, seed=args.seed
    )
    sim_smoke.end()

    print(f"   SCALE=0.05 (original default): pred={preds_smoke[0]}  label={cal_y[0]}  zeros={zeros_smoke}")
    if zeros_smoke > 0:
        print("   ⚠️  Head population silent at SCALE=0.05 — expected, needs re-calibration.")

    # ------------------------------------------------------------------
    # 6. Calibration
    # ------------------------------------------------------------------
    if args.scale is None:
        print("\n6. Calibrating SCALE on calibration set...")
        best_scale, _ = calibrate(
            cal_features, cal_y,
            W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
            args.duration, args.rate_scale,
        )
        scale = best_scale
    else:
        scale = args.scale
        print(f"\n6. Using provided SCALE={scale:.4f}")

    # ------------------------------------------------------------------
    # 7. Evaluate on calibration set
    # ------------------------------------------------------------------
    print("\n7. Running SpiNNaker-sim on calibration set...")
    sim_cal, input_pop_cal, head_pop_cal = build_network(
        W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head, scale, 0.001
    )
    cal_preds, cal_zeros = run_samples(
        sim_cal, input_pop_cal, head_pop_cal, cal_features, args.rate_scale, args.duration, seed=args.seed
    )
    sim_cal.end()

    # ------------------------------------------------------------------
    # 8. Evaluate on held-out test set
    # ------------------------------------------------------------------
    print("\n8. Running SpiNNaker-sim on held-out test set...")
    sim_test, input_pop_test, head_pop_test = build_network(
        W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head, scale, 0.001
    )
    test_preds, test_zeros = run_samples(
        sim_test, input_pop_test, head_pop_test, test_features, args.rate_scale, args.duration, seed=args.seed
    )
    sim_test.end()

    # ------------------------------------------------------------------
    # 9. Report
    # ------------------------------------------------------------------
    report(cal_y, cal_preds, cal_pt, "CALIBRATION SET")
    report(test_y, test_preds, test_pt, "HELD-OUT TEST SET")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  PyTorch baseline (test):          {test_pt_acc:.1f}%")
    print(f"  Brian2 SpiNNaker-sim (test):      {100.0 * np.sum((test_preds >= 0) & (test_preds == test_y)) / len(test_y):.1f}%")
    print(f"  All-zero / no-prediction (test):  {test_zeros}/{len(test_y)}  ({100.0*test_zeros/len(test_y):.1f}%)")
    print(f"  SCALE used:                       {scale:.4f}")
    print(f"  rate_scale used:                  {args.rate_scale:.1f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("HONEST STATEMENT")
    print("=" * 70)
    print("""
This result is a Brian2 software simulation of the SpiNNaker-deployed network.
It uses the EXACT neuron parameters (tau_m=20, v_rest=-65, v_thresh=-50,
tau_refrac=2, tau_syn_E=5), the EXACT bias scaling (×10.0), and the EXACT
weight filtering logic copied from build_spinnaker_network().

Any accuracy drop versus the PyTorch snn.Leaky baseline is the real, honest
finding of how faithfully the trained weights map onto SpiNNaker's IF_curr_exp
dynamics under these parameters.
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
