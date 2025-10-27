# MoE Expert Analysis Scripts

Collection of scripts for analyzing MoE (Mixture of Experts) expert selection patterns from llama-server logs.

## Prerequisites

```bash
pip install pandas numpy matplotlib seaborn
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

### Data Format

Expected CSV format from llama-server:
```csv
layer_id,expert_ids
3,52 40 85 110 79 113 60 130
3,32 115 108 7 141 124 11 3
```

Where:
- `layer_id`: MoE layer number
- `expert_ids`: Space-separated list of selected expert IDs (top-k)

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
