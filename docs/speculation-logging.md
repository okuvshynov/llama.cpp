# Speculative Decoding Logging

Detailed logging for speculative decoding analysis and parameter tuning.

## Usage

```bash
# Run server with speculation logging enabled
llama-server \
  -m model.gguf \
  -md draft.gguf \
  --draft 32 \
  --draft-p-min 0.5 \
  --spec-log-file /tmp/spec.jsonl

# Analyze the logs
python scripts/spec_log_analyze.py /tmp/spec.jsonl
```

## Log Format

JSON Lines format, one object per speculation round:

```json
{
  "ts_ms": 1703001234567,
  "slot_id": 0,
  "prompt_pos": 125,
  "round_id": 42,
  "draft": {
    "n": 5,
    "stop_reason": "p_min",
    "tokens": [
      {"pos": 0, "id": 1234, "p": 0.85, "entropy": 0.45, "top_k": [[1234, 0.85], [5678, 0.08]]}
    ]
  },
  "verify": {
    "n_accepted": 3,
    "rej_pos": 3,
    "target_id": 7890
  },
  "timing": {
    "t_draft_us": 12500,
    "t_verify_us": 45000
  }
}
```

## Fields

| Field | Description |
|-------|-------------|
| `draft.n` | Number of tokens drafted |
| `draft.stop_reason` | Why drafting stopped: `p_min`, `n_max`, or `complete` |
| `draft.tokens[].p` | Probability of selected token |
| `draft.tokens[].entropy` | Entropy of top-k distribution |
| `draft.tokens[].top_k` | Top-10 candidates as `[token_id, probability]` pairs |
| `verify.n_accepted` | How many draft tokens were accepted |
| `verify.rej_pos` | Position of first rejection (-1 if all accepted) |
| `verify.target_id` | Token the target model selected |
| `timing.t_draft_us` | Time spent generating draft tokens (microseconds) |
| `timing.t_verify_us` | Time spent verifying with target model (microseconds) |

## Analysis Output

```
SPECULATIVE DECODING LOG ANALYSIS
======================================================================
OVERALL STATISTICS
----------------------------------------------------------------------
  Total Rounds:            128
  Total Drafted:          1024
  Total Accepted:          669
  Acceptance Rate:       65.32%

POSITION-WISE ANALYSIS
----------------------------------------------------------------------
Pos   Drafted   Accepted   Rate %     Avg Entropy  Avg Prob
0     128       115        89.84      0.3421       0.8932
1     128       102        79.69      0.4512       0.8234
2     125       89         71.20      0.5234       0.7654
```
