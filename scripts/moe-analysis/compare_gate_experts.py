#!/usr/bin/env python3
"""
Compare expert selection using same gate weights but different inputs.

Tests if different model quantizations (which produce different gate inputs)
result in different expert selections when using the same gate weights.

This is useful for understanding if quantizing the main model (attention and
expert weights) affects which experts get selected, even though the gate
weights themselves remain the same.

Usage:
    python compare_gate_experts.py \\
        --weight layer_20_gate_weight.npy \\
        --bias layer_20_gate_bias.npy \\
        --inputs-a inputs_f16.txt \\
        --inputs-b inputs_q4.txt \\
        --top-k 8

Input format: One vector per line, space-separated floats (no header)

Reports precision, recall, and ranking metrics for expert selection.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from collections import defaultdict


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for array x."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid values for array x."""
    return 1 / (1 + np.exp(-x))


def load_gate_inputs(filepath: str) -> List[np.ndarray]:
    """
    Load gate inputs from a file.

    Format: One vector per line, space-separated floats.
    The first 3 columns (if present, like "gate_input,layer,token,") are ignored.

    Returns:
        List of gate input vectors
    """
    gate_inputs = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Check if line starts with "gate_input," and skip those columns
                if line.startswith('gate_input,'):
                    # Format: gate_input,layer_id,token_id,values...
                    parts = line.split(',', 3)
                    if len(parts) == 4:
                        values_str = parts[3]
                    else:
                        continue
                else:
                    # Plain format: values only
                    values_str = line

                values = np.array([float(x) for x in values_str.split()])
                gate_inputs.append(values)

            except Exception as e:
                print(f"Warning: Skipping line {line_num}: {e}", file=sys.stderr)
                continue

    return gate_inputs


def compute_expert_logits(gate_input: np.ndarray,
                          weight: np.ndarray,
                          bias: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute expert logits from gate input.

    Args:
        gate_input: [n_embd] input vector
        weight: [n_embd, n_expert] weight matrix
        bias: [n_expert] bias vector (optional)

    Returns:
        [n_expert] logits
    """
    logits = gate_input @ weight  # [n_embd] @ [n_embd, n_expert] = [n_expert]

    if bias is not None:
        logits = logits + bias

    return logits


def get_top_k_experts(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Get top-k expert indices from logits.

    Args:
        logits: [n_expert] logit values
        k: number of top experts to select

    Returns:
        [k] indices of top experts (sorted by logit value, descending)
    """
    # argsort in descending order
    return np.argsort(logits)[::-1][:k]


def compute_metrics(original_experts: np.ndarray,
                    candidate_experts: np.ndarray,
                    k: int,
                    all_logits_b: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute precision, recall, and ranking metrics.

    Args:
        original_experts: [k] indices of experts selected by original weights
        candidate_experts: [k] indices of experts selected by candidate weights
        k: number of top experts
        all_logits_b: [n_expert] all logits from candidate input (optional)

    Returns:
        Dictionary of metrics
    """
    original_set = set(original_experts)
    candidate_set = set(candidate_experts)

    # Precision: what fraction of candidate's selections are correct?
    # Recall: what fraction of original's selections did candidate find?
    intersection = len(original_set & candidate_set)
    precision = intersection / k if k > 0 else 0.0
    recall = intersection / k if k > 0 else 0.0  # Same as precision for top-k

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Exact match: all experts are the same (order doesn't matter)
    exact_match = 1.0 if original_set == candidate_set else 0.0

    # Coverage metric: How many top-N from B do we need to cover top-K from A?
    coverage_k = k  # Default to k if we can't compute
    if all_logits_b is not None:
        # Get all experts sorted by logits from B
        all_experts_b_sorted = np.argsort(all_logits_b)[::-1]  # Descending order

        # Find how many we need to take from B to cover all experts from A
        for n in range(k, len(all_experts_b_sorted) + 1):
            top_n_from_b = set(all_experts_b_sorted[:n])
            if original_set.issubset(top_n_from_b):
                coverage_k = n
                break
        else:
            # If we can't cover all, set to total number of experts
            coverage_k = len(all_experts_b_sorted)

    # Ranking metrics
    # Kendall's Tau: correlation between rankings
    # For experts that appear in both, compute rank correlation
    common_experts = original_set & candidate_set
    if len(common_experts) > 1:
        # Get ranks in both lists
        original_ranks = {expert: i for i, expert in enumerate(original_experts)}
        candidate_ranks = {expert: i for i, expert in enumerate(candidate_experts)}

        # Compute Kendall's Tau for common experts
        concordant = 0
        discordant = 0
        common_list = list(common_experts)
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                exp_i, exp_j = common_list[i], common_list[j]
                orig_order = original_ranks[exp_i] < original_ranks[exp_j]
                cand_order = candidate_ranks[exp_i] < candidate_ranks[exp_j]
                if orig_order == cand_order:
                    concordant += 1
                else:
                    discordant += 1

        total_pairs = concordant + discordant
        kendall_tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0
    else:
        kendall_tau = 1.0 if len(common_experts) == 1 else 0.0

    # Top-1 match: does the top expert match?
    top1_match = 1.0 if original_experts[0] == candidate_experts[0] else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': exact_match,
        'top1_match': top1_match,
        'kendall_tau': kendall_tau,
        'intersection_size': intersection,
        'coverage_k': coverage_k,
    }


def compare_gate_inputs(inputs_a: List[np.ndarray],
                        inputs_b: List[np.ndarray],
                        weight: np.ndarray,
                        bias: Optional[np.ndarray],
                        top_k: int = 8,
                        gating_func: str = 'softmax') -> Dict:
    """
    Compare expert selection using same gate weights but different inputs.

    Args:
        inputs_a: List of gate input vectors from model A
        inputs_b: List of gate input vectors from model B
        weight: Gate weight matrix
        bias: Gate bias vector (optional)
        top_k: Number of top experts to select
        gating_func: Gating function ('softmax' or 'sigmoid')

    Returns:
        Dictionary with per-sample and aggregate metrics
    """
    results = {
        'per_sample': [],
        'aggregate': {},
    }

    # Ensure same number of samples
    n_samples = min(len(inputs_a), len(inputs_b))
    if len(inputs_a) != len(inputs_b):
        print(f"Warning: Input files have different lengths ({len(inputs_a)} vs {len(inputs_b)}). "
              f"Using first {n_samples} samples.", file=sys.stderr)

    # Accumulate metrics across all samples
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)

    for idx in range(n_samples):
        input_a = inputs_a[idx]
        input_b = inputs_b[idx]

        # Compute logits for both inputs using same gate weights
        logits_a = compute_expert_logits(input_a, weight, bias)
        logits_b = compute_expert_logits(input_b, weight, bias)

        # Get top-k experts
        experts_a = get_top_k_experts(logits_a, top_k)
        experts_b = get_top_k_experts(logits_b, top_k)

        # Compute metrics (pass logits_b for coverage metric)
        metrics = compute_metrics(experts_a, experts_b, top_k, all_logits_b=logits_b)

        # Store per-sample results
        sample_result = {
            'sample_idx': idx,
            'experts_a': experts_a.tolist(),
            'experts_b': experts_b.tolist(),
            'metrics': metrics,
        }
        results['per_sample'].append(sample_result)

        # Accumulate for aggregate stats
        for key, value in metrics.items():
            metric_sums[key] += value
            metric_counts[key] += 1

    # Compute aggregate statistics
    if n_samples > 0:
        for key in metric_sums.keys():
            results['aggregate'][f'{key}_mean'] = metric_sums[key] / metric_counts[key]

        # Compute standard deviations
        for key in metric_sums.keys():
            values = [sample['metrics'][key] for sample in results['per_sample']]
            results['aggregate'][f'{key}_std'] = np.std(values)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare expert selection using same gate weights but different inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compare_gate_experts.py \\
    --weight layer_20_gate_weight.npy \\
    --bias layer_20_gate_bias.npy \\
    --inputs-a inputs_f16.txt \\
    --inputs-b inputs_q4.txt \\
    --top-k 8

Input file format (one vector per line):
  gate_input,20,0,0.123 -0.456 0.789 ... (with header - will be stripped)
  0.123 -0.456 0.789 ...                 (without header - plain values)
        """
    )

    parser.add_argument('--weight', '-w', required=True,
                        help='Path to gate weight (.npy)')
    parser.add_argument('--bias', '-b', default=None,
                        help='Path to gate bias (.npy, optional)')
    parser.add_argument('--inputs-a', '-a', required=True,
                        help='Path to inputs from model A (e.g., F16)')
    parser.add_argument('--inputs-b', '-B', required=True,
                        help='Path to inputs from model B (e.g., Q4)')
    parser.add_argument('--top-k', '-k', type=int, default=8,
                        help='Number of top experts to select (default: 8)')
    parser.add_argument('--gating-func', choices=['softmax', 'sigmoid'], default='softmax',
                        help='Gating function (default: softmax)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print per-sample results')

    args = parser.parse_args()

    print("=" * 80)
    print("Expert Selection Comparison")
    print("=" * 80)
    print()

    # Load gate weights
    print(f"Loading gate weights:")
    print(f"  Weight: {args.weight}")
    weight = np.load(args.weight)
    print(f"    Shape: {weight.shape}")

    bias = None
    if args.bias:
        print(f"  Bias: {args.bias}")
        bias = np.load(args.bias)
        print(f"    Shape: {bias.shape}")
    print()

    # Load inputs from both models
    print(f"Loading inputs from model A: {args.inputs_a}")
    inputs_a = load_gate_inputs(args.inputs_a)
    print(f"  Loaded {len(inputs_a)} samples")
    if len(inputs_a) > 0:
        print(f"  Vector dimension: {inputs_a[0].shape[0]}")

    print(f"Loading inputs from model B: {args.inputs_b}")
    inputs_b = load_gate_inputs(args.inputs_b)
    print(f"  Loaded {len(inputs_b)} samples")
    if len(inputs_b) > 0:
        print(f"  Vector dimension: {inputs_b[0].shape[0]}")
    print()

    if len(inputs_a) == 0 or len(inputs_b) == 0:
        print("Error: No inputs found in one or both files", file=sys.stderr)
        sys.exit(1)

    # Check gate input dimension matches weight dimension
    n_embd_input = inputs_a[0].shape[0]
    n_embd_weight = weight.shape[0]
    if n_embd_input != n_embd_weight:
        print(f"Error: Gate input dimension ({n_embd_input}) doesn't match weight dimension ({n_embd_weight})",
              file=sys.stderr)
        sys.exit(1)

    # Run comparison
    print(f"Comparing expert selection (top-{args.top_k})...")
    print(f"Using same gate weights, different inputs from two model quantizations")
    print()

    results = compare_gate_inputs(
        inputs_a, inputs_b,
        weight, bias,
        args.top_k,
        args.gating_func
    )

    # Print per-sample results if verbose
    if args.verbose:
        print("=" * 80)
        print("Per-Sample Results")
        print("=" * 80)
        for sample in results['per_sample']:
            print(f"\nSample {sample['sample_idx']}:")
            print(f"  Model A experts: {sample['experts_a']}")
            print(f"  Model B experts: {sample['experts_b']}")
            print(f"  Metrics:")
            for key, value in sample['metrics'].items():
                print(f"    {key:20s}: {value:.4f}")
        print()

    # Print aggregate statistics
    print("=" * 80)
    print("Aggregate Statistics")
    print("=" * 80)
    print()

    agg = results['aggregate']
    n_samples = len(results['per_sample'])
    print(f"Samples analyzed: {n_samples}")
    print(f"Top-K: {args.top_k}")
    print()

    print(f"Precision (mean ± std):     {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}")
    print(f"Recall (mean ± std):        {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}")
    print(f"F1 Score (mean ± std):      {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")
    print(f"Exact Match Rate:           {agg['exact_match_mean']:.4f}")
    print(f"Top-1 Match Rate:           {agg['top1_match_mean']:.4f}")
    print(f"Kendall's Tau (mean ± std): {agg['kendall_tau_mean']:.4f} ± {agg['kendall_tau_std']:.4f}")
    print(f"Coverage K (mean ± std):    {agg['coverage_k_mean']:.1f} ± {agg['coverage_k_std']:.1f}")
    print()

    print("Interpretation:")
    print(f"  - {agg['exact_match_mean']*100:.1f}% of samples have identical expert selections")
    print(f"  - {agg['top1_match_mean']*100:.1f}% of samples have the same top-1 expert")
    print(f"  - Average {agg['intersection_size_mean']:.1f}/{args.top_k} experts match per sample")
    print(f"  - Need top-{agg['coverage_k_mean']:.1f} from model B to cover top-{args.top_k} from model A")
    print()
    print("Coverage K interpretation:")
    print(f"  - Perfect match: Coverage K = {args.top_k} (top-{args.top_k} from B covers top-{args.top_k} from A)")
    print(f"  - Higher values indicate more ranking shift between models")
    print()
    print("This shows how different model quantizations (e.g., F16 vs Q4) affect")
    print("which experts get selected, even with the same gate weights.")


if __name__ == '__main__':
    main()
