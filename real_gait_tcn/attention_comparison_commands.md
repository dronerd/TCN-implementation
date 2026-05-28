# TCN Attention Comparison Commands

Use this file for `real_gait_tcn/compare_tcn_attention.py`, which compares baseline TCN against TCN + attention.

Run all commands from the project root:

```bash
cd "/Users/yutokuroki/Desktop/TCN implementation"
source venv/bin/activate
```

Outputs are stored under:

```text
results/attention_comparison/
├── json/
├── plots/
└── notes/
```

Prediction plots use:

- Black: truth value, `hip_moment_l`
- Blue: TCN without attention
- Red: TCN + attention

## 1. Small Test

Practical all-user comparison test. This is the first attention comparison run to use.

Estimated time on CPU: about 10-25 minutes.

```bash
python real_gait_tcn/compare_tcn_attention.py \
  --num-users 0 \
  --splits 5 \
  --epochs 2 \
  --batch-size 32 \
  --sequence-stride 500 \
  --latency-batches 20 \
  --latency-iterations 10 \
  --plot-samples 300 \
  --plot-width 18 \
  --plot-height 4 \
  --plot-dir results/attention_comparison/plots/small_test \
  --results-file results/attention_comparison/json/cv_small_test.json
```

## 2. Large Test

Heavier all-user comparison test with more windows and more training.

Estimated time on CPU: about 2-5 hours.

```bash
python real_gait_tcn/compare_tcn_attention.py \
  --num-users 0 \
  --splits 5 \
  --epochs 5 \
  --batch-size 32 \
  --sequence-stride 100 \
  --latency-batches 20 \
  --latency-iterations 10 \
  --plot-samples 500 \
  --plot-width 24 \
  --plot-height 4 \
  --plot-dir results/attention_comparison/plots/large_test \
  --results-file results/attention_comparison/json/cv_large_test.json
```

## 3. Full Test

Full all-user comparison training using every overlapping sequence window.

Warning: this can take multiple days on CPU, and possibly longer than a week.

```bash
python real_gait_tcn/compare_tcn_attention.py \
  --num-users 0 \
  --splits 5 \
  --epochs 20 \
  --batch-size 32 \
  --sequence-stride 1 \
  --latency-batches 20 \
  --latency-iterations 10 \
  --plot-samples 300 \
  --plot-width 18 \
  --plot-height 4 \
  --plot-dir results/attention_comparison/plots/full_test \
  --results-file results/attention_comparison/json/cv_full_test.json
```
