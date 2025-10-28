#!/usr/bin/env python3
"""
Analyze gate input differences between consecutive MoE layers.

Usage:
    cat moe_expert_selection.log | grep gate_input > gates_inputs.log
    python gate_input_analysis.py gates_inputs.log

Analyzes token_id=0 entries (generated tokens) and computes distance metrics
between consecutive layers (20->21, 21->22, 22->23, 23->24).
"""

import sys
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr


def parse_gate_input_line(line: str) -> Tuple[int, int, np.ndarray]:
    """
    Parse a gate_input log line.

    Format: gate_input,layer_id,token_id,value1 value2 value3 ...

    Returns:
        (layer_id, token_id, values_array)
    """
    parts = line.strip().split(',', 3)
    if len(parts) != 4 or parts[0] != 'gate_input':
        raise ValueError(f"Invalid line format: {line[:100]}")

    layer_id = int(parts[1])
    token_id = int(parts[2])
    values = np.array([float(x) for x in parts[3].split()])

    return layer_id, token_id, values


def load_gate_inputs(filepath: str) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Load all gate inputs from file.

    Returns:
        Dictionary mapping (layer_id, token_id) -> gate_input_vector
    """
    gate_inputs = {}

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                layer_id, token_id, values = parse_gate_input_line(line)
                gate_inputs[(layer_id, token_id)] = values
            except Exception as e:
                print(f"Warning: Skipping line {line_num}: {e}", file=sys.stderr)
                continue

    return gate_inputs


def compute_distances(vec1: np.ndarray, vec2: np.ndarray) -> Dict[str, float]:
    """
    Compute various distance/similarity metrics between two vectors.

    Returns:
        Dictionary with metric names and values
    """
    metrics = {}

    # Cosine similarity (1 - cosine distance)
    metrics['cosine_similarity'] = 1 - cosine(vec1, vec2)
    metrics['cosine_distance'] = cosine(vec1, vec2)

    # Euclidean distance
    metrics['euclidean'] = euclidean(vec1, vec2)

    # L1 (Manhattan) distance
    metrics['l1'] = np.sum(np.abs(vec1 - vec2))

    # L2 normalized (divide by vector dimension)
    metrics['l2_normalized'] = np.linalg.norm(vec1 - vec2) / np.sqrt(len(vec1))

    # Pearson correlation
    if len(vec1) > 1:
        try:
            metrics['pearson_corr'], _ = pearsonr(vec1, vec2)
        except:
            metrics['pearson_corr'] = np.nan
    else:
        metrics['pearson_corr'] = np.nan

    # Max absolute difference
    metrics['max_abs_diff'] = np.max(np.abs(vec1 - vec2))

    # Mean absolute difference
    metrics['mean_abs_diff'] = np.mean(np.abs(vec1 - vec2))

    return metrics


def analyze_generation_tokens(gate_inputs: Dict[Tuple[int, int], np.ndarray],
                               layers: List[int] = [20, 21, 22, 23, 24]) -> None:
    """
    Analyze gate inputs for generated tokens (token_id == 0).

    For each occurrence of token_id=0 (each generated token), compute
    distances between consecutive layers.
    """
    # Find all unique positions where we have token_id=0
    # Each position represents a generated token
    positions = set()
    for (layer_id, token_id) in gate_inputs.keys():
        if token_id == 0:
            positions.add((layer_id, token_id))

    # Group by occurrence (each time we see the full layer sequence)
    # We need to identify which occurrence of token_id=0 for each layer

    # Create a list of (index, layer_id, gate_input) for token_id=0
    token_0_entries = []
    layer_counts = {layer: 0 for layer in layers}

    # We need to track which "batch" each entry belongs to
    # Assumption: entries come in order, and each layer appears once per batch

    # Better approach: scan sequentially and group into batches
    batches = []
    current_batch = {}
    last_layer = layers[-1]

    # Read entries in order from the dictionary (but dict is unordered)
    # We need to parse the file again to maintain order
    # Or we can organize by counting occurrences

    # Let's use a different approach: find all occurrences by counting
    # For each layer, count how many token_id=0 entries exist
    occurrences = {}
    for layer in layers:
        count = 0
        # We need to know the order, so we'll have to reparse
        # For now, let's just count
        for (lid, tid) in gate_inputs.keys():
            if lid == layer and tid == 0:
                count += 1
        occurrences[layer] = count

    # Minimum count across all layers is the number of complete batches
    num_batches = min(occurrences.values())

    print(f"Found {num_batches} complete generation steps (token_id=0 across all layers)")
    print(f"Occurrences per layer: {occurrences}")
    print()

    # Since we can't reliably determine order from dict, we need to reparse
    # Let's create a simpler version that just computes stats across all pairs

    print("=" * 80)
    print("Layer-to-Layer Distance Analysis (token_id=0 only)")
    print("=" * 80)
    print()

    # For each consecutive layer pair, collect all distances
    for i in range(len(layers) - 1):
        layer1, layer2 = layers[i], layers[i + 1]

        print(f"Layer {layer1} -> Layer {layer2}")
        print("-" * 40)

        # We need to match up corresponding token_id=0 entries
        # Since we can't determine order from dict alone, let's just report what we can

        # Check if we have any data
        has_layer1 = any((lid, 0) in gate_inputs for lid in [layer1])
        has_layer2 = any((lid, 0) in gate_inputs for lid in [layer2])

        if not (has_layer1 and has_layer2):
            print("  No data available for this layer pair\n")
            continue

        # For now, just compute distance for the first occurrence we find
        vec1 = gate_inputs.get((layer1, 0))
        vec2 = gate_inputs.get((layer2, 0))

        if vec1 is not None and vec2 is not None:
            metrics = compute_distances(vec1, vec2)

            print(f"  Vector dimension: {len(vec1)}")
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
            print(f"  Cosine distance:   {metrics['cosine_distance']:.6f}")
            print(f"  Euclidean dist:    {metrics['euclidean']:.4f}")
            print(f"  L2 normalized:     {metrics['l2_normalized']:.6f}")
            print(f"  L1 distance:       {metrics['l1']:.4f}")
            print(f"  Pearson corr:      {metrics['pearson_corr']:.6f}")
            print(f"  Max abs diff:      {metrics['max_abs_diff']:.6f}")
            print(f"  Mean abs diff:     {metrics['mean_abs_diff']:.6f}")
        else:
            print("  Missing data for one or both layers")

        print()


def analyze_generation_tokens_from_file(filepath: str,
                                        layers: List[int] = [20, 21, 22, 23, 24]) -> None:
    """
    Analyze gate inputs by parsing file sequentially to maintain order.

    This properly handles multiple generation steps.
    """
    # Parse file sequentially and group into generation steps
    generation_steps = []
    current_step = {}

    with open(filepath, 'r') as f:
        for line in f:
            try:
                layer_id, token_id, values = parse_gate_input_line(line)

                # Only process token_id == 0 (generated tokens)
                if token_id != 0:
                    continue

                # Check if this layer is in our target layers
                if layer_id not in layers:
                    continue

                # If we see layer 20 again, it means a new generation step
                if layer_id == layers[0] and current_step:
                    generation_steps.append(current_step)
                    current_step = {}

                current_step[layer_id] = values

            except Exception as e:
                continue

    # Add the last step
    if current_step:
        generation_steps.append(current_step)

    print(f"Found {len(generation_steps)} generation steps")
    print()

    # Analyze each generation step
    for step_idx, step_data in enumerate(generation_steps):
        # Skip if we don't have all layers
        if not all(layer in step_data for layer in layers):
            continue

        print(f"{'=' * 80}")
        print(f"Generation Step #{step_idx + 1}")
        print(f"{'=' * 80}")
        print()

        # Compute distances between consecutive layers
        for i in range(len(layers) - 1):
            layer1, layer2 = layers[i], layers[i + 1]

            vec1 = step_data[layer1]
            vec2 = step_data[layer2]

            metrics = compute_distances(vec1, vec2)

            print(f"Layer {layer1} -> {layer2}:")
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
            print(f"  Cosine distance:   {metrics['cosine_distance']:.6f}")
            print(f"  Euclidean dist:    {metrics['euclidean']:.4f}")
            print(f"  L2 normalized:     {metrics['l2_normalized']:.6f}")
            print(f"  Pearson corr:      {metrics['pearson_corr']:.6f}")
            print(f"  Mean abs diff:     {metrics['mean_abs_diff']:.6f}")
            print()

        print()

    # Compute aggregate statistics across all generation steps
    if len(generation_steps) > 1:
        print(f"{'=' * 80}")
        print(f"Aggregate Statistics Across {len(generation_steps)} Generation Steps")
        print(f"{'=' * 80}")
        print()

        # Collect metrics for each layer pair across all steps
        for i in range(len(layers) - 1):
            layer1, layer2 = layers[i], layers[i + 1]

            all_metrics = {
                'cosine_similarity': [],
                'cosine_distance': [],
                'euclidean': [],
                'l2_normalized': [],
                'pearson_corr': [],
                'mean_abs_diff': []
            }

            for step_data in generation_steps:
                if layer1 in step_data and layer2 in step_data:
                    metrics = compute_distances(step_data[layer1], step_data[layer2])
                    for key in all_metrics.keys():
                        if not np.isnan(metrics[key]):
                            all_metrics[key].append(metrics[key])

            print(f"Layer {layer1} -> {layer2} (mean ± std):")
            for metric_name, values in all_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  {metric_name:20s}: {mean_val:.6f} ± {std_val:.6f}")
            print()


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <gates_inputs.log>", file=sys.stderr)
        print(f"\nPreprocess with: cat moe_expert_selection.log | grep gate_input > gates_inputs.log", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]

    print("Gate Input Layer Distance Analysis")
    print("=" * 80)
    print()

    # Analyze with sequential parsing to maintain generation step order
    analyze_generation_tokens_from_file(filepath)


if __name__ == '__main__':
    main()
