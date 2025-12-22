#!/usr/bin/env python
"""
Verify MLP dump by running forward pass on CPU and comparing to ground truth.

MLP forward pass (SwiGLU):
    gate = input @ W_gate
    up = input @ W_up
    output = (silu(gate) * up) @ W_down

Where silu(x) = x * sigmoid(x)
"""

import numpy as np
import sys
import os

def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))

def mlp_forward(input_tensor, gate_weight, up_weight, down_weight):
    """
    Run MLP forward pass.

    Args:
        input_tensor: (M, hidden_size)
        gate_weight: (hidden_size, intermediate_size)
        up_weight: (hidden_size, intermediate_size)
        down_weight: (intermediate_size, hidden_size)

    Returns:
        output: (M, hidden_size)
    """
    # gate = input @ W_gate  -> (M, intermediate)
    gate = input_tensor @ gate_weight

    # up = input @ W_up  -> (M, intermediate)
    up = input_tensor @ up_weight

    # gated = silu(gate) * up  -> (M, intermediate)
    gated = silu(gate) * up

    # output = gated @ W_down  -> (M, hidden)
    output = gated @ down_weight

    return output

def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Loading data from: {dump_dir}")
    print()

    # Load weights
    gate_weight = np.load(os.path.join(dump_dir, "layer0_ffn_gate.npy"))
    up_weight = np.load(os.path.join(dump_dir, "layer0_ffn_up.npy"))
    down_weight = np.load(os.path.join(dump_dir, "layer0_ffn_down.npy"))

    print(f"gate_weight: {gate_weight.shape}")
    print(f"up_weight:   {up_weight.shape}")
    print(f"down_weight: {down_weight.shape}")
    print()

    # Load activations
    # ffn_norm = MLP input (after RMSNorm)
    # ffn_out = MLP output (after down projection, before residual)
    ffn_inp = np.load(os.path.join(dump_dir, "ffn_norm-0.npy"))
    ffn_out_expected = np.load(os.path.join(dump_dir, "ffn_out-0.npy"))

    print(f"ffn_norm (MLP input):   {ffn_inp.shape}")
    print(f"ffn_out (MLP output):   {ffn_out_expected.shape}")
    print()

    # Run forward pass
    print("Running MLP forward pass on CPU...")
    ffn_out_computed = mlp_forward(ffn_inp, gate_weight, up_weight, down_weight)

    print(f"ffn_out (computed):     {ffn_out_computed.shape}")
    print()

    # Compare
    diff = ffn_out_computed - ffn_out_expected
    max_abs_err = np.abs(diff).max()
    mean_abs_err = np.abs(diff).mean()

    # Relative error (avoiding division by zero)
    abs_expected = np.abs(ffn_out_expected)
    rel_err = np.abs(diff) / np.maximum(abs_expected, 1e-8)
    max_rel_err = rel_err.max()

    print("=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"Max absolute error:  {max_abs_err:.6e}")
    print(f"Mean absolute error: {mean_abs_err:.6e}")
    print(f"Max relative error:  {max_rel_err:.6e}")
    print()

    # Show some sample values
    print("Sample values (first 5 elements of first token):")
    print(f"  Expected: {ffn_out_expected[0, :5]}")
    print(f"  Computed: {ffn_out_computed[0, :5]}")
    print(f"  Diff:     {diff[0, :5]}")
    print()

    # Pass/fail
    tolerance = 1e-3
    if max_abs_err < tolerance:
        print(f"PASS: Max error {max_abs_err:.2e} < tolerance {tolerance:.0e}")
        return 0
    else:
        print(f"FAIL: Max error {max_abs_err:.2e} >= tolerance {tolerance:.0e}")

        # Find worst element
        worst_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"  Worst at index {worst_idx}:")
        print(f"    Expected: {ffn_out_expected[worst_idx]:.6f}")
        print(f"    Computed: {ffn_out_computed[worst_idx]:.6f}")
        print(f"    Diff:     {diff[worst_idx]:.6f}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
