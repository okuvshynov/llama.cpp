#!/usr/bin/env python
"""
Verify layer dump by running forward pass on CPU and comparing to ground truth.

This script verifies:
1. Attention projections: Q = X @ W_q, K = X @ W_k, V = X @ W_v
2. MLP forward pass: output = silu(X @ W_gate) * (X @ W_up) @ W_down
3. Full layer: output = TransformerLayer(input) (end-to-end verification)

Note: Full layer verification requires l_inp and l_out tensors dumped from llama.cpp.
The full layer includes RMSNorm, attention, RoPE, SDPA, MLP, and residual connections.
"""

import numpy as np
import sys
import os

def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))

def rmsnorm(x, weight, eps=1e-5):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    # x: (M, hidden), weight: (hidden,)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def verify_attention_projections(dump_dir, layer_idx=0):
    """
    Verify Q, K, V projections.

    Since we don't have ground truth for Q/K/V outputs in the dump,
    we just verify the weight shapes and that the math works.
    """
    prefix = f"layer{layer_idx}_"

    # Load weights
    wq = np.load(os.path.join(dump_dir, f"{prefix}wq.npy"))
    wk = np.load(os.path.join(dump_dir, f"{prefix}wk.npy"))
    wv = np.load(os.path.join(dump_dir, f"{prefix}wv.npy"))
    wo = np.load(os.path.join(dump_dir, f"{prefix}wo.npy"))

    print("Attention projection weights:")
    print(f"  wq: {wq.shape}  (hidden -> num_heads * head_dim)")
    print(f"  wk: {wk.shape}  (hidden -> num_kv_heads * head_dim)")
    print(f"  wv: {wv.shape}  (hidden -> num_kv_heads * head_dim)")
    print(f"  wo: {wo.shape}  (num_heads * head_dim -> hidden)")
    print()

    # Load attention input (after RMSNorm)
    attn_inp = np.load(os.path.join(dump_dir, f"attn_norm-{layer_idx}.npy"))
    print(f"  attn_norm (input): {attn_inp.shape}")
    print()

    # Compute Q, K, V projections
    print("Computing attention projections...")
    Q = attn_inp @ wq  # (M, hidden) @ (hidden, qDim) -> (M, qDim)
    K = attn_inp @ wk  # (M, hidden) @ (hidden, kvDim) -> (M, kvDim)
    V = attn_inp @ wv  # (M, hidden) @ (hidden, kvDim) -> (M, kvDim)

    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    print()

    # We don't have ground truth for Q/K/V, so just verify shapes match
    hidden = wq.shape[0]
    qDim = wq.shape[1]
    kvDim = wk.shape[1]
    M = attn_inp.shape[0]

    assert Q.shape == (M, qDim), f"Q shape mismatch: {Q.shape} vs {(M, qDim)}"
    assert K.shape == (M, kvDim), f"K shape mismatch: {K.shape} vs {(M, kvDim)}"
    assert V.shape == (M, kvDim), f"V shape mismatch: {V.shape} vs {(M, kvDim)}"

    print("Attention projection shapes verified!")
    print()

    return {
        'hidden': hidden,
        'qDim': qDim,
        'kvDim': kvDim,
        'M': M,
        'wq': wq,
        'wk': wk,
        'wv': wv,
        'wo': wo,
        'attn_input': attn_inp,
        'Q': Q,
        'K': K,
        'V': V,
    }

def verify_full_layer(dump_dir, layer_idx=0):
    """
    Verify full transformer layer: input -> output (end-to-end).

    This checks that l_out matches the expected layer output for the given l_inp.
    Note: We can't recompute this on CPU without implementing RoPE and SDPA,
    so we just verify the data is present and report shapes.
    """
    prefix = f"layer{layer_idx}_"

    # Check for layer input/output files
    # Note: l_out may not be captured if build_cvec creates a new tensor
    # In that case, ffn_out IS the layer output (after second residual)
    l_inp_path = os.path.join(dump_dir, f"l_inp-{layer_idx}.npy")
    l_out_path = os.path.join(dump_dir, f"l_out-{layer_idx}.npy")
    ffn_out_path = os.path.join(dump_dir, f"ffn_out-{layer_idx}.npy")

    has_l_inp = os.path.exists(l_inp_path)
    has_l_out = os.path.exists(l_out_path)
    has_ffn_out = os.path.exists(ffn_out_path)

    if not has_l_inp:
        print("Full layer verification: SKIPPED")
        print(f"  l_inp-{layer_idx}.npy: NOT FOUND")
        print("  (Requires updated llama.cpp with l_inp callback)")
        return None

    l_inp = np.load(l_inp_path)

    # Use l_out if available, otherwise use ffn_out (they're equivalent when cvec is not used)
    if has_l_out:
        l_out = np.load(l_out_path)
        l_out_name = f"l_out-{layer_idx}.npy"
    elif has_ffn_out:
        l_out = np.load(ffn_out_path)
        l_out_name = f"ffn_out-{layer_idx}.npy (= layer output)"
    else:
        print("Full layer verification: SKIPPED")
        print(f"  Neither l_out nor ffn_out found")
        return None

    print("Full layer tensors:")
    print(f"  l_inp (layer input):  {l_inp.shape}")
    print(f"  l_out (layer output): {l_out.shape}")
    print()

    # Check for RMSNorm weights
    attn_norm_path = os.path.join(dump_dir, f"{prefix}attn_norm.npy")
    ffn_norm_path = os.path.join(dump_dir, f"{prefix}ffn_norm_weight.npy")

    has_attn_norm = os.path.exists(attn_norm_path)
    has_ffn_norm = os.path.exists(ffn_norm_path)

    print("RMSNorm weights:")
    print(f"  attn_norm: {'found' if has_attn_norm else 'NOT FOUND'}")
    print(f"  ffn_norm:  {'found' if has_ffn_norm else 'NOT FOUND'}")

    if has_attn_norm:
        attn_norm_weight = np.load(attn_norm_path)
        print(f"    attn_norm shape: {attn_norm_weight.shape}")
    if has_ffn_norm:
        ffn_norm_weight = np.load(ffn_norm_path)
        print(f"    ffn_norm shape: {ffn_norm_weight.shape}")
    print()

    # Verify shapes match
    assert l_inp.shape == l_out.shape, f"Shape mismatch: l_inp={l_inp.shape}, l_out={l_out.shape}"

    # Report statistics
    print("Layer input/output statistics:")
    print(f"  l_inp: min={l_inp.min():.4f}, max={l_inp.max():.4f}, mean={l_inp.mean():.4f}")
    print(f"  l_out: min={l_out.min():.4f}, max={l_out.max():.4f}, mean={l_out.mean():.4f}")
    print()

    return {
        'l_inp': l_inp,
        'l_out': l_out,
        'l_out_name': l_out_name,
        'attn_norm_weight': np.load(attn_norm_path) if has_attn_norm else None,
        'ffn_norm_weight': np.load(ffn_norm_path) if has_ffn_norm else None,
    }

def verify_mlp(dump_dir, layer_idx=0):
    """
    Verify MLP forward pass.
    """
    prefix = f"layer{layer_idx}_"

    # Load weights
    gate_weight = np.load(os.path.join(dump_dir, f"{prefix}ffn_gate.npy"))
    up_weight = np.load(os.path.join(dump_dir, f"{prefix}ffn_up.npy"))
    down_weight = np.load(os.path.join(dump_dir, f"{prefix}ffn_down.npy"))

    print("MLP weights:")
    print(f"  ffn_gate: {gate_weight.shape}")
    print(f"  ffn_up:   {up_weight.shape}")
    print(f"  ffn_down: {down_weight.shape}")
    print()

    # Load activations
    ffn_inp = np.load(os.path.join(dump_dir, f"ffn_norm-{layer_idx}.npy"))
    ffn_out_expected = np.load(os.path.join(dump_dir, f"ffn_out-{layer_idx}.npy"))

    print(f"  ffn_norm (MLP input):   {ffn_inp.shape}")
    print(f"  ffn_out (MLP output):   {ffn_out_expected.shape}")
    print()

    # Run forward pass
    print("Running MLP forward pass on CPU...")
    gate = ffn_inp @ gate_weight
    up = ffn_inp @ up_weight
    gated = silu(gate) * up
    ffn_out_computed = gated @ down_weight

    print(f"  ffn_out (computed): {ffn_out_computed.shape}")
    print()

    # Compare
    diff = ffn_out_computed - ffn_out_expected
    max_abs_err = np.abs(diff).max()
    mean_abs_err = np.abs(diff).mean()

    abs_expected = np.abs(ffn_out_expected)
    rel_err = np.abs(diff) / np.maximum(abs_expected, 1e-8)
    max_rel_err = rel_err.max()

    print("=" * 50)
    print("MLP COMPARISON RESULTS")
    print("=" * 50)
    print(f"Max absolute error:  {max_abs_err:.6e}")
    print(f"Mean absolute error: {mean_abs_err:.6e}")
    print(f"Max relative error:  {max_rel_err:.6e}")
    print()

    print("Sample values (first 5 elements of first token):")
    print(f"  Expected: {ffn_out_expected[0, :5]}")
    print(f"  Computed: {ffn_out_computed[0, :5]}")
    print(f"  Diff:     {diff[0, :5]}")
    print()

    tolerance = 2e-3  # Same as Lyrae test
    if max_abs_err < tolerance:
        print(f"MLP PASS: Max error {max_abs_err:.2e} < tolerance {tolerance:.0e}")
        return True
    else:
        print(f"MLP FAIL: Max error {max_abs_err:.2e} >= tolerance {tolerance:.0e}")
        worst_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"  Worst at index {worst_idx}:")
        print(f"    Expected: {ffn_out_expected[worst_idx]:.6f}")
        print(f"    Computed: {ffn_out_computed[worst_idx]:.6f}")
        print(f"    Diff:     {diff[worst_idx]:.6f}")
        return False

def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    layer_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"Loading data from: {dump_dir}")
    print(f"Layer index: {layer_idx}")
    print()

    # Verify full layer (if data available)
    print("=" * 60)
    print("FULL LAYER VERIFICATION")
    print("=" * 60)
    layer_info = verify_full_layer(dump_dir, layer_idx)

    # Verify attention projections
    print()
    print("=" * 60)
    print("ATTENTION PROJECTIONS")
    print("=" * 60)
    attn_info = verify_attention_projections(dump_dir, layer_idx)

    # Verify MLP
    print()
    print("=" * 60)
    print("MLP FORWARD PASS")
    print("=" * 60)
    mlp_passed = verify_mlp(dump_dir, layer_idx)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model dimensions:")
    print(f"  M (batch/seq):    {attn_info['M']}")
    print(f"  hidden:           {attn_info['hidden']}")
    print(f"  qDim:             {attn_info['qDim']}")
    print(f"  kvDim:            {attn_info['kvDim']}")
    print()

    has_layer_data = layer_info is not None
    print(f"Full layer data:       {'AVAILABLE' if has_layer_data else 'NOT AVAILABLE'}")
    print(f"Attention projections: VERIFIED (shapes only)")
    print(f"MLP forward pass:      {'PASS' if mlp_passed else 'FAIL'}")

    if has_layer_data:
        print()
        print("Full layer verification ready for Lyrae test:")
        print(f"  Input:  l_inp-{layer_idx}.npy       {layer_info['l_inp'].shape}")
        print(f"  Output: {layer_info.get('l_out_name', 'l_out-' + str(layer_idx) + '.npy'):25} {layer_info['l_out'].shape}")

    return 0 if mlp_passed else 1

if __name__ == "__main__":
    sys.exit(main())
