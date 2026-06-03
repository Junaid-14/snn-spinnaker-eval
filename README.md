# SNN SpiNNaker Evaluation

> Software simulation of a trained spiking neural network on SpiNNaker's
> IF_curr_exp neuron dynamics, cross-validated against Brian2/PyNN.
> **This is a software simulation of the SpiNNaker neuron model —
> not a result from SpiNNaker hardware.**

## Overview

This repository evaluates whether a trained SNN (spiking neural network)
survives mapping to SpiNNaker's IF_curr_exp neuron dynamics. The core
question: *if we take PyTorch snn.Leaky weights and deploy them on a
SpiNNaker-like IF_curr_exp model (tau_m=20 ms, v_thresh=−50 mV, etc.),
how much accuracy is lost?*

**Headline result:** 93.86% ± 0.15% (SpiNNaker sim, 5 seeds, 2000 samples)
vs 94.90% (PyTorch baseline) on N-MNIST — a **1.04 percentage point**
systematic gap, consistent across all seeds.

## Methodology

- **Pipeline:** PyTorch conv frontend → FC backbone (1600→1024→512→10) →
  Poisson rate encoding → IF_curr_exp discrete-time simulation
- **SpiNNaker IF_curr_exp parameters** (from `build_spinnaker_network()`):
  - tau_m = 20 ms, v_rest = v_reset = −65 mV, v_thresh = −50 mV
  - tau_refrac = 2 ms, tau_syn_E/I = 5 ms, cm = 1 nF
  - Bias = trained_bias × 10.0 (no SCALE factor)
  - Synaptic delay = 1 ms
  - SCALE = 7.0 (calibrated once, locked)
- **Bug found and corrected:** The original simulator used a
  constant-current approximation (`v_factor = 0.9754 mV/nA`); the corrected
  version uses the exact `iaf_psc_exp` propagator cross-factor
  (`0.8833 mV/nA`), matching Brian2/PyNN's "exact" integration method.
- **Validation:** Brian2/PyNN cross-check on 20 stratified reference samples
  at SCALE=10 with identical Poisson spike arrays — **20/20 argmax predictions
  matched**. Residual spike-count differences (3–7 spikes total across the 10
  head neurons) are attributable to discrete-time refractory quantization.
  Agreement at SCALE=10 (higher firing, the more stringent case) implies
  agreement at the reported SCALE=7.

## Results

| Statistic | Value |
|-----------|-------|
| **SpiNNaker sim mean accuracy** | **93.86%** |
| **Std (5 seeds)** | **0.15%** |
| **Seed range (min–max)** | **93.75%–94.15%** |
| PyTorch baseline (deterministic) | 94.90% |
| Agreement with PyTorch | 95.32% ± 0.18% |
| All-zero / no-prediction samples | **0 / 10,000** (5 seeds × 2000) |

**Per-seed breakdown:**

| Seed | Accuracy | Agreement w/ PyTorch |
|------|----------|---------------------|
| 42 | 94.15% | 95.25% |
| 123 | 93.75% | 95.05% |
| 456 | 93.75% | 95.35% |
| 789 | 93.85% | 95.60% |
| 999 | 93.80% | 95.35% |

**Mean confusion matrix** (SpiNNaker sim, averaged over 5 seeds, 2000 samples = 200/class):

| Pred \ True | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------------|---|---|---|---|---|---|---|---|---|---|
| **0** | 194.8 | 0.2 | 2.2 | 0.8 | 0.0 | 0.2 | 4.0 | 0.8 | 3.2 | 0.2 |
| **1** | 0.6 | 195.2 | 1.4 | 0.0 | 1.4 | 0.4 | 1.6 | 2.6 | 1.8 | 1.0 |
| **2** | 0.0 | 0.2 | 186.8 | 1.4 | 1.4 | 0.4 | 0.6 | 1.4 | 1.2 | 0.4 |
| **3** | 0.0 | 0.0 | 3.0 | 189.4 | 0.2 | 5.2 | 0.0 | 1.0 | 5.6 | 1.8 |
| **4** | 0.0 | 0.6 | 3.6 | 0.0 | 190.6 | 0.8 | 2.4 | 0.4 | 1.4 | 1.2 |
| **5** | 0.6 | 0.0 | 0.0 | 4.2 | 0.2 | 188.4 | 5.2 | 0.4 | 3.8 | 2.2 |
| **6** | 2.2 | 2.2 | 0.2 | 0.0 | 1.0 | 1.0 | 184.8 | 0.0 | 0.6 | 0.0 |
| **7** | 0.0 | 0.2 | 0.4 | 0.2 | 0.2 | 0.0 | 0.0 | 184.6 | 1.6 | 1.8 |
| **8** | 1.0 | 1.2 | 1.4 | 2.8 | 0.8 | 2.4 | 1.4 | 1.4 | 178.6 | 7.4 |
| **9** | 0.8 | 0.2 | 1.0 | 1.2 | 4.2 | 1.2 | 0.0 | 7.4 | 2.2 | 184.0 |

The 1.04 pp gap is **systematic** (std 0.15%), not seed noise — it reflects
the cost of Poisson rate-coding and IF_curr_exp dynamics.

## Files

| File | Description |
|------|-------------|
| `test_spinnaker_discrete_exact.py` | Fast batched PyTorch discrete-time simulator of the full SpiNNaker IF_curr_exp network. Calibrates SCALE, runs full evaluation, reports accuracy and confusion matrix. |
| `test_spinnaker_brian2_exact.py` | Brian2/PyNN event-based simulator (slow for dense layers; kept for reference). Replicates `build_spinnaker_network()` exactly using PyNN native projections. |
| `verify_tiny.py` | Tiny-network (5→3→2) variant comparison script. Tests multiple discretization variants against Brian2/PyNN to find the correct update equations. |
| `verify_full.py` | Full-network verification: runs the corrected PyTorch sim and raw Brian2 on 20 stratified samples with identical spike arrays, comparing per-neuron counts and argmax predictions. |

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

**Note:** `sPyNNaker` requires a SpiNNaker board or virtual-board setup.
For pure software verification, `pyNN.brian2` (included via `brian2`)
is sufficient.

## Usage

### 1. Run the main evaluation (fast PyTorch simulator)

Single run (one seed, quick check):

```bash
python test_spinnaker_discrete_exact.py \
    --checkpoint path/to/your/checkpoint.pt \
    --cal-samples 50 \
    --test-samples 2000 \
    --scale 7.0 \
    --rate-scale 10.0 \
    --duration 100.0 \
    --seed 42
```

Reproduce the reported result (mean over 5 seeds, 2000 samples):

```bash
for SEED in 42 123 456 789 999; do
  python test_spinnaker_discrete_exact.py \
      --checkpoint path/to/your/checkpoint.pt \
      --cal-samples 50 \
      --test-samples 2000 \
      --scale 7.0 \
      --rate-scale 10.0 \
      --duration 100.0 \
      --seed $SEED
done
```
(Then average the five reported accuracies.)

### 2. Verify against Brian2 on a tiny reference network

```bash
python verify_tiny.py
```

### 3. Verify against Brian2 on 20 real N-MNIST samples

```bash
python verify_full.py \
    --checkpoint path/to/your/checkpoint.pt
```

### 4. Run the slow Brian2/PyNN simulator (reference only)

```bash
python test_spinnaker_brian2_exact.py \
    --checkpoint path/to/your/checkpoint.pt \
    --cal-samples 50 \
    --test-samples 500
```

## Limitations

- **Software simulation only:** Results are from floating-point simulation
  of the SpiNNaker neuron model, not from actual SpiNNaker hardware.
- **Fixed-point effects:** Hardware-specific fixed-point quantization of
  weights and membrane potentials is not captured.
- **Checkpoint not included:** You must provide your own trained N-MNIST
  SNN checkpoint. The scripts expect a checkpoint with keys matching
  `backbones.nmnist.fc*.weight/bias` and `heads.nmnist.1.weight/bias`.
- **Data loader dependency:** The scripts import `Dataloaders.nmnist_loader`
  from the parent project. You may need to adapt the data loading code
  to your own N-MNIST dataset source.
