# MoE Expert Analysis Scripts

Collection of scripts for analyzing MoE (Mixture of Experts) expert selection patterns from llama-server logs.

## Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Scripts

### 1. `expert_frequency.py`

Analyzes expert selection frequency per layer.

**Usage:**
```bash
python expert_frequency.py moe_expert_selection.log
```

**Output:**
- Per-layer statistics: top/bottom experts by frequency
- Global statistics across all layers
- Helps identify load balancing and expert utilization

### 2. `gate_input_analysis.py`

Analyzes gate input differences between consecutive MoE layers for generated tokens.

**Usage:**
```bash
# Preprocess to extract gate inputs
cat moe_expert_selection.log | grep gate_input > gates_inputs.log

# Run analysis
python gate_input_analysis.py gates_inputs.log
```

**Output:**
- Per-generation-step distance metrics between consecutive layers (20→21, 21→22, etc.)
- Aggregate statistics across all generation steps
- Multiple distance metrics:
  - Cosine similarity/distance
  - Euclidean distance
  - L2 normalized distance
  - Pearson correlation
  - Mean absolute difference

**What it analyzes:**
- Only processes `token_id=0` entries (generated tokens, not prompt)
- Computes how gate inputs change between consecutive layers
- Helps understand how token representations evolve through MoE layers

### 3. `extract_gate_tensors.py`

Extracts MoE gate (router) weights and bias tensors from a GGUF model.

**Usage:**
```bash
# Extract tensors for a single layer
python extract_gate_tensors.py model.gguf 20

# Extract to specific directory
python extract_gate_tensors.py model.gguf 20 --output-dir tensors/

# Extract multiple layers
for layer in 20 21 22 23 24; do
  python extract_gate_tensors.py model.gguf $layer --output-dir tensors/
done
```

**Extracts:**
- `blk.{layer}.ffn_gate_inp.weight`: Router weight matrix `[n_embd, n_expert]`
- `blk.{layer}.exp_probs_b.bias`: Expert selection bias `[n_expert]` (if present)

**Output formats:**
- NumPy arrays (`.npy`) - always saved
- PyTorch tensors (`.pt`) - saved if PyTorch is available

**Use case:**
Allows you to reproduce the expert selection locally using the logged gate inputs:
```python
import numpy as np

# Load extracted tensors
weight = np.load('layer_20_gate_weight.npy')  # [n_embd, n_expert]
bias = np.load('layer_20_gate_bias.npy')      # [n_expert]

# Load gate input from log
gate_input = ...  # [n_embd] from gates_inputs.log

# Compute logits (what the model does internally)
logits = gate_input @ weight  # [n_expert]
logits = logits + bias        # Add bias (if present)

# Apply gating function (softmax or sigmoid depending on model)
probs = softmax(logits)
```

## Data Formats

### Expert Selection Log Format

```csv
layer_id,expert_ids
3,52 40 85 110 79 113 60 130
3,32 115 108 7 141 124 11 3
```

Where:
- `layer_id`: MoE layer number
- `expert_ids`: Space-separated list of selected expert IDs (top-k)

### Gate Input Log Format

```csv
gate_input,layer_id,token_id,value1 value2 value3 ...
gate_input,20,0,0.123 -0.456 0.789 ... (n_embd values)
gate_input,21,0,-0.234 0.567 -0.890 ... (n_embd values)
```

Where:
- `layer_id`: MoE layer number (20-24 in current implementation)
- `token_id`: Token position in batch (0 = first token in generation)
- `values`: Space-separated gate input values (n_embd floats, e.g., 5120 values)

## Example Findings

From uniform distribution (~0.6% per expert):
- Good load balancing across 160 experts
- No routing collapse or dead experts
- Suggests diverse token representations

## Future Analysis Ideas

- Expert co-occurrence patterns (which experts are selected together?)
- Layer-wise expert specialization
- Temporal patterns (expert usage over time)
- Correlation with prompt types (code vs text vs math)
