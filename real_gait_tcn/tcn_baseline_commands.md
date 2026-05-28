# TCN Baseline Commands

Use this file for `real_gait_tcn/tcn_baseline.py`, which trains and evaluates only the baseline TCN without attention.

Run all commands from the project root:

```bash
cd "/Users/yutokuroki/Desktop/TCN implementation"
source venv/bin/activate
```

Outputs are stored under:

```text
results/tcn_baseline/
├── json/
└── plots/
```

## 1. Small Test

Practical all-user baseline test. This is the first run to use.

Estimated time on CPU: about 5-15 minutes.

```bash
python real_gait_tcn/tcn_baseline.py \
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
  --plot-dir results/tcn_baseline/plots/small_test \
  --results-file results/tcn_baseline/json/tcn_small_test.json
```

## 2. Large Test

Heavier all-user baseline test with more windows and more training.

Estimated time on CPU: about 1-3 hours.

```bash
python real_gait_tcn/tcn_baseline.py \
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
  --plot-dir results/tcn_baseline/plots/large_test \
  --results-file results/tcn_baseline/json/tcn_large_test.json
```

## 3. Full Test

Full all-user baseline training using every overlapping sequence window.

Warning: this can take multiple days on CPU.

```bash
python real_gait_tcn/tcn_baseline.py \
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
  --plot-dir results/tcn_baseline/plots/full_test \
  --results-file results/tcn_baseline/json/tcn_full_test.json
```
