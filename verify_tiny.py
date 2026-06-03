#!/usr/bin/env python3
"""
Verify PyTorch discrete-time simulator against Brian2/PyNN on a TINY network.
"""

import numpy as np
import torch
import pyNN.brian2 as sim
from brian2 import ms, defaultclock


def run_pytorch_variant(spike_data, W1, b1, W2, b2, scale, duration_ms, variant):
    n_samples, n_steps, n_pre = spike_data.shape
    n_mid = W1.shape[0]
    n_post = W2.shape[0]

    W1t = torch.from_numpy(W1).float() * scale
    W1t = W1t * (torch.abs(W1t) > 0.001)
    b1t = torch.from_numpy(b1).float() * 10.0

    W2t = torch.from_numpy(W2).float() * scale
    W2t = W2t * (torch.abs(W2t) > 0.001)
    b2t = torch.from_numpy(b2).float() * 10.0

    dt = 1.0
    alpha_v = np.exp(-dt / 20.0)
    alpha_ie = np.exp(-dt / 5.0)
    v_factor = 20.0 * (1.0 - alpha_v)
    cross_factor = (20.0 * 5.0 / 15.0) * (alpha_v - alpha_ie)

    refrac_steps = {'A': 2, 'D': 2, 'H': 3, 'I': 3}[variant]

    results = []
    for s in range(n_samples):
        input_spikes = torch.from_numpy(spike_data[s]).float()

        v_mid = torch.full((n_mid,), -65.0)
        ie_mid = torch.zeros(n_mid)
        r_mid = torch.zeros(n_mid, dtype=torch.int32)

        v_post = torch.full((n_post,), -65.0)
        ie_post = torch.zeros(n_post)
        r_post = torch.zeros(n_post, dtype=torch.int32)
        post_acc = torch.zeros(n_post, dtype=torch.int32)

        for t in range(n_steps):
            # --- FC1 ---
            if variant in ('A', 'D'):
                ie_mid = ie_mid * alpha_ie + input_spikes[t] @ W1t.T
                ie_temp = ie_mid
            else:  # H, I
                ie_temp = ie_mid + input_spikes[t] @ W1t.T
                ie_mid = ie_temp * alpha_ie

            if variant in ('A', 'H'):
                v_mid = torch.where(r_mid == 0, -65.0 + (v_mid + 65.0) * alpha_v + (ie_temp + b1t) * v_factor, v_mid)
            else:  # D, I
                v_mid = torch.where(r_mid == 0, -65.0 + (v_mid + 65.0) * alpha_v + ie_temp * cross_factor + b1t * v_factor, v_mid)

            spike_mid = (r_mid == 0) & (v_mid >= -50.0)
            v_mid = torch.where(spike_mid, torch.full_like(v_mid, -65.0), v_mid)
            r_mid = torch.where(spike_mid, torch.full_like(r_mid, refrac_steps, dtype=torch.int32), r_mid)
            r_mid = torch.clamp(r_mid - 1, min=0)

            # --- FC2 ---
            if variant in ('A', 'D'):
                ie_post = ie_post * alpha_ie + spike_mid.float() @ W2t.T
                ie_temp_post = ie_post
            else:
                ie_temp_post = ie_post + spike_mid.float() @ W2t.T
                ie_post = ie_temp_post * alpha_ie

            if variant in ('A', 'H'):
                v_post = torch.where(r_post == 0, -65.0 + (v_post + 65.0) * alpha_v + (ie_temp_post + b2t) * v_factor, v_post)
            else:
                v_post = torch.where(r_post == 0, -65.0 + (v_post + 65.0) * alpha_v + ie_temp_post * cross_factor + b2t * v_factor, v_post)

            spike_post = (r_post == 0) & (v_post >= -50.0)
            v_post = torch.where(spike_post, torch.full_like(v_post, -65.0), v_post)
            r_post = torch.where(spike_post, torch.full_like(r_post, refrac_steps, dtype=torch.int32), r_post)
            r_post = torch.clamp(r_post - 1, min=0)

            post_acc += spike_post.to(torch.int32)

        results.append(post_acc.numpy().copy())
    return np.stack(results)


def run_brian2_tiny(spike_data_list, W1, b1, W2, b2, scale, duration_ms):
    n_samples = len(spike_data_list)
    n_pre = W1.shape[1]
    n_mid = W1.shape[0]
    n_post = W2.shape[0]
    results = []

    for s in range(n_samples):
        indices, times = spike_data_list[s]

        defaultclock.dt = 1.0 * ms
        sim.setup(timestep=1.0, min_delay=1.0)

        input_pop = sim.Population(n_pre, sim.SpikeSourceArray(spike_times=[]))
        if len(indices) > 0:
            input_pop.brian2_group.set_spikes(indices, times * ms)

        mid_pop = sim.Population(n_mid, sim.IF_curr_exp(
            tau_m=20.0, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
            tau_refrac=2.0, tau_syn_E=5.0, tau_syn_I=5.0, cm=1.0, i_offset=0.0))
        mid_pop.initialize(v=-65.0)
        mid_pop.record(["spikes"])

        post_pop = sim.Population(n_post, sim.IF_curr_exp(
            tau_m=20.0, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
            tau_refrac=2.0, tau_syn_E=5.0, tau_syn_I=5.0, cm=1.0, i_offset=0.0))
        post_pop.initialize(v=-65.0)
        post_pop.record(["spikes"])

        def build_proj(pre, post, W, b):
            conn = []
            for post_i in range(W.shape[0]):
                for pre_i in range(W.shape[1]):
                    w = float(W[post_i, pre_i]) * scale
                    if abs(w) > 0.001:
                        conn.append((pre_i, post_i, w, 1.0))
            if conn:
                sim.Projection(pre, post, sim.FromListConnector(conn), sim.StaticSynapse(), receptor_type="excitatory")
            for i in range(W.shape[0]):
                post[i:i+1].set(i_offset=float(b[i]) * 10.0)

        build_proj(input_pop, mid_pop, W1, b1)
        build_proj(mid_pop, post_pop, W2, b2)

        sim.run(duration_ms)

        data = post_pop.get_data("spikes")
        counts = np.array([len(st) for st in data.segments[-1].spiketrains])
        results.append(counts)

        sim.end()

    return np.stack(results)


def main():
    print("=" * 70)
    print("TINY NETWORK VERIFICATION")
    print("=" * 70)

    rng = np.random.default_rng(123)
    n_pre, n_mid, n_post = 5, 3, 2
    W1 = rng.normal(0, 0.3, (n_mid, n_pre)).astype(np.float64)
    b1 = rng.normal(0.02, 0.05, n_mid).astype(np.float64)
    W2 = rng.normal(0, 0.3, (n_post, n_mid)).astype(np.float64)
    b2 = rng.normal(0.02, 0.05, n_post).astype(np.float64)

    scale = 20.0
    duration = 50.0
    n_samples = 5

    spike_data = np.zeros((n_samples, int(duration), n_pre), dtype=bool)
    spike_data_list = []
    for s in range(n_samples):
        mask = rng.random((int(duration), n_pre)) < 0.8
        spike_data[s] = mask
        indices = []
        times = []
        for nid in range(n_pre):
            ts = np.where(mask[:, nid])[0]
            for t in ts:
                indices.append(nid)
                times.append(float(t))
        spike_data_list.append((np.array(indices, dtype=np.int32), np.array(times, dtype=np.float64)))

    print(f"\nNetwork: {n_pre} -> {n_mid} -> {n_post}, SCALE={scale}, duration={duration}ms")

    print("\nRunning Brian2 sim...")
    b2_counts = run_brian2_tiny(spike_data_list, W1, b1, W2, b2, scale, duration)

    variants = {
        'A': 'spikes after decay, v_factor, refrac=2',
        'D': 'spikes after decay, cross-factor, refrac=2',
        'H': 'spikes before decay, v_factor, refrac=3',
        'I': 'spikes before decay, cross-factor, refrac=3',
    }

    print("\n" + "-" * 70)
    print(f"{'Var':>3} | {'Description':>40} | {'Match':>5} | {'MaxD':>4} | Counts")
    print("-" * 70)

    best_variant = None
    best_diff = float('inf')

    for var, desc in variants.items():
        pt_counts = run_pytorch_variant(spike_data, W1, b1, W2, b2, scale, duration, var)
        match = "Y" if np.array_equal(pt_counts, b2_counts) else "N"
        maxdiff = int(np.abs(pt_counts - b2_counts).max())
        counts_str = " ".join(f"[{c[0]:2d} {c[1]:2d}]" for c in pt_counts)
        print(f"{var:>3} | {desc:>40} | {match:>5} | {maxdiff:>4d} | {counts_str}")
        if maxdiff < best_diff:
            best_diff = maxdiff
            best_variant = var

    print("-" * 70)
    print(f"B2  | {'Brian2':>40} |       |      | {' '.join(f'[{c[0]:2d} {c[1]:2d}]' for c in b2_counts)}")
    print("-" * 70)

    if best_diff == 0:
        print(f"\n✓ PERFECT MATCH: Variant {best_variant} matches Brian2/PyNN IF_curr_exp exactly.")
    else:
        print(f"\n⚠ Closest: Variant {best_variant} with max diff = {best_diff} spikes.")

    print("=" * 70)


if __name__ == "__main__":
    main()
