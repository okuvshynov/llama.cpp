#!/usr/bin/env python3
"""
Extract MoE gate (router) weights and bias tensors from a GGUF model.

Usage:
    python extract_gate_tensors.py <model.gguf> <layer_id> [--output-dir output/]

Extracts:
- blk.{layer}.ffn_gate_inp.weight: Router weight matrix [n_embd, n_expert]
- blk.{layer}.exp_probs_b.bias: Expert selection bias [n_expert] (if present)

Saves as NumPy arrays (.npy) and optionally PyTorch tensors (.pt).
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add gguf-py to path
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize


def find_tensor_by_name(reader: GGUFReader, tensor_name: str):
    """Find a tensor by exact name match."""
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            return tensor
    return None


def dequantize_tensor(tensor) -> np.ndarray:
    """
    Dequantize a tensor to float32.

    If already F32/F16, return as-is. Otherwise use gguf's dequantize function.
    """
    # If already float type, just return the data
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return tensor.data.astype(np.float32)
    elif tensor.tensor_type == GGMLQuantizationType.F16:
        return tensor.data.astype(np.float32)

    # Otherwise, dequantize
    try:
        # The dequantize function takes the quantized data and returns float32
        dequantized = dequantize(tensor.data, tensor.tensor_type)
        return dequantized.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to dequantize {tensor.name}: {e}", file=sys.stderr)
        print(f"Attempting to use raw data...", file=sys.stderr)
        return tensor.data.astype(np.float32)


def extract_gate_tensors(model_path: str, layer_id: int, output_dir: str = "."):
    """
    Extract gate weight and bias tensors for a specific layer.

    Args:
        model_path: Path to GGUF model file
        layer_id: Layer number to extract (e.g., 7 for blk.7.*)
        output_dir: Directory to save extracted tensors
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    reader = GGUFReader(model_path)

    # Construct tensor names
    weight_name = f"blk.{layer_id}.ffn_gate_inp.weight"
    bias_name = f"blk.{layer_id}.exp_probs_b.bias"

    print(f"\nSearching for tensors:")
    print(f"  - {weight_name}")
    print(f"  - {bias_name} (optional)")
    print()

    # Find weight tensor
    weight_tensor = find_tensor_by_name(reader, weight_name)
    if weight_tensor is None:
        print(f"Error: Weight tensor '{weight_name}' not found in model", file=sys.stderr)
        print(f"\nAvailable tensors matching 'ffn_gate_inp':", file=sys.stderr)
        for tensor in reader.tensors:
            if 'ffn_gate_inp' in tensor.name:
                print(f"  {tensor.name}", file=sys.stderr)
        sys.exit(1)

    print(f"Found weight tensor: {weight_name}")
    print(f"  Shape: {weight_tensor.shape}")
    print(f"  Type: {weight_tensor.tensor_type.name}")
    print(f"  Elements: {weight_tensor.n_elements}")
    print(f"  Bytes: {weight_tensor.n_bytes}")

    # Dequantize weight
    print(f"  Dequantizing...")
    weight_data = dequantize_tensor(weight_tensor)

    # Reshape to [n_embd, n_expert] if needed
    if len(weight_tensor.shape) > 2:
        # Sometimes shape is [n_embd, n_expert, 1, 1]
        weight_data = weight_data.reshape(weight_tensor.shape[0], weight_tensor.shape[1])

    print(f"  Final shape: {weight_data.shape}")

    # Save weight as numpy
    weight_npy_path = output_path / f"layer_{layer_id}_gate_weight.npy"
    np.save(weight_npy_path, weight_data)
    print(f"  Saved: {weight_npy_path}")

    # Try to save as PyTorch if available
    try:
        import torch
        weight_torch = torch.from_numpy(weight_data)
        weight_pt_path = output_path / f"layer_{layer_id}_gate_weight.pt"
        torch.save(weight_torch, weight_pt_path)
        print(f"  Saved: {weight_pt_path}")
    except ImportError:
        print(f"  (PyTorch not available, skipping .pt format)")

    # Find bias tensor (optional)
    bias_tensor = find_tensor_by_name(reader, bias_name)
    if bias_tensor is None:
        print(f"\nBias tensor '{bias_name}' not found (this is optional)")
        print(f"Note: Not all models have expert selection bias")
    else:
        print(f"\nFound bias tensor: {bias_name}")
        print(f"  Shape: {bias_tensor.shape}")
        print(f"  Type: {bias_tensor.tensor_type.name}")
        print(f"  Elements: {bias_tensor.n_elements}")

        # Dequantize bias
        print(f"  Dequantizing...")
        bias_data = dequantize_tensor(bias_tensor)

        # Flatten if needed
        if len(bias_tensor.shape) > 1:
            bias_data = bias_data.flatten()

        print(f"  Final shape: {bias_data.shape}")

        # Save bias as numpy
        bias_npy_path = output_path / f"layer_{layer_id}_gate_bias.npy"
        np.save(bias_npy_path, bias_data)
        print(f"  Saved: {bias_npy_path}")

        # Try to save as PyTorch
        try:
            import torch
            bias_torch = torch.from_numpy(bias_data)
            bias_pt_path = output_path / f"layer_{layer_id}_gate_bias.pt"
            torch.save(bias_torch, bias_pt_path)
            print(f"  Saved: {bias_pt_path}")
        except ImportError:
            pass

    print(f"\nExtraction complete!")
    print(f"\nTo use with logged gate inputs:")
    print(f"  import numpy as np")
    print(f"  weight = np.load('{weight_npy_path}')")
    if bias_tensor:
        print(f"  bias = np.load('{bias_npy_path}')")
    print(f"  gate_input = ...  # Load from gates_inputs.log")
    print(f"  logits = gate_input @ weight  # Matrix multiply [n_embd] @ [n_embd, n_expert] = [n_expert]")
    if bias_tensor:
        print(f"  logits = logits + bias  # Add bias")
    print(f"  probs = softmax(logits)  # or sigmoid, depending on model")

    return {
        'weight': weight_data,
        'bias': bias_data if bias_tensor else None,
        'layer_id': layer_id,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract MoE gate weights and bias from GGUF model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract layer 20 tensors
  python extract_gate_tensors.py model.gguf 20

  # Extract to specific directory
  python extract_gate_tensors.py model.gguf 20 --output-dir extracted_tensors/

  # Extract multiple layers
  for layer in 20 21 22 23 24; do
    python extract_gate_tensors.py model.gguf $layer --output-dir tensors/
  done
        """
    )

    parser.add_argument('model_path', type=str,
                        help='Path to GGUF model file')
    parser.add_argument('layer_id', type=int,
                        help='Layer ID to extract (e.g., 20 for blk.20.*)')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='Output directory for extracted tensors (default: current dir)')

    args = parser.parse_args()

    extract_gate_tensors(args.model_path, args.layer_id, args.output_dir)


if __name__ == '__main__':
    main()
