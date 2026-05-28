# Real Gait TCN Experiments

This project predicts left hip moment, `hip_moment_l`, from wearable gait sensor data using Temporal Convolutional Networks.

## Dataset Origin And Attribution

The gait dataset used by this project is not original data collected for this repository. It comes from the EPIC Lab / Young Lab dataset released on Dryad for the Science Robotics study by Molinaro, Kang, and Young.

The recommended workflow is:

1. Run the baseline TCN first.
2. Inspect its validation loss, latency, and prediction plots.
3. Then run the TCN vs TCN + attention comparison.
4. Decide whether attention improves prediction enough to justify its extra inference cost.

The full dataset is read from:

```text
data/doi_10_5061_dryad_8kprr4xsv__v20240321
```

Original dataset source:

- Molinaro, Dean (2024). *Estimating human joint moments unifies exoskeleton control and reduces user effort* [Dataset]. Dryad. https://doi.org/10.5061/dryad.8kprr4xsv

Associated paper:

- Molinaro, Dean D.; Kang, Inseung; Young, Aaron J. (2024). *Estimating human joint moments unifies exoskeleton control, reducing user effort*. Science Robotics, 9(88), eadi8852. https://doi.org/10.1126/scirobotics.adi8852

## Setup

From the project root:

```bash
source venv/bin/activate
```

If dependencies are missing:

```bash
pip install -r real_gait_tcn/requirements.txt
```

## 1. Start With The Baseline TCN

The baseline script is:

```text
real_gait_tcn/tcn_baseline.py
```

It trains only `TCNWithoutAttention` from `model.py`.

It reports:

- User-stratified validation loss
- TCN inference latency
- TCN throughput
- Prediction plots with truth vs TCN prediction

Prediction plot colors:

- Black: true `hip_moment_l`
- Blue: TCN prediction

Run the Small Test first:

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

For more baseline TCN commands, see:

```text
real_gait_tcn/tcn_baseline_commands.md
```

## 2. Why Add Attention?

Gait signals are temporal and cyclic, but different users do not move in exactly the same way. Even for the same task, the timing of important events can shift between participants because of differences in stride, biomechanics, sensor placement, and movement strategy.

A TCN learns temporal patterns with 1D convolutions. In this project, the TCN uses dilated convolutions, so each later layer sees a wider region of the input sequence. This is useful for time-series prediction because the model can learn short and broader motion patterns.

Attention adds a second idea: instead of using only the final TCN representation, the model can compare timesteps with each other and emphasize the parts of the sequence that are most informative for the prediction.

For hip moment prediction, attention may help when:

- The important cue is not always at the same exact sample index.
- Some parts of the 100-timestep window are more informative than others.
- Different participants have slightly different gait timing.
- The model needs to relate earlier and later motion patterns inside the same window.

Attention is not automatically better. It adds computation and can increase latency. That is why this project compares both accuracy and inference time.

## 3. Run The Attention Comparison

The comparison script is:

```text
real_gait_tcn/compare_tcn_attention.py
```

It trains and compares:

- `TCNWithoutAttention`
- `TCNWithAttention`

Both models are defined in:

```text
real_gait_tcn/model.py
```

The comparison reports:

- Validation loss without attention
- Validation loss with attention
- Percent improvement from attention
- Inference latency for both models
- Throughput for both models
- Attention latency overhead
- Prediction plots with truth, TCN, and TCN + attention

Prediction plot colors:

- Black: true `hip_moment_l`
- Blue: TCN without attention
- Red: TCN + attention

Run the Small Test first:

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

For Small, Large, and Full comparison commands, see:

```text
real_gait_tcn/attention_comparison_commands.md
```

## 4. How The Implementation Is Organized

The important files are:

```text
real_gait_tcn/
├── model.py
├── tcn_baseline.py
├── compare_tcn_attention.py
├── tcn_baseline_commands.md
├── attention_comparison_commands.md
└── requirements.txt
```

Results are organized under:

```text
results/
├── tcn_baseline/
│   ├── json/
│   └── plots/
└── attention_comparison/
    ├── json/
    ├── plots/
    └── notes/
```

See `results/README.md` for the folder convention.

### `model.py`

This file contains the neural network definitions.

`TemporalBlock` is the basic TCN block:

- 1D convolution
- ReLU
- Dropout
- Another 1D convolution
- Residual connection

`TCNWithoutAttention` uses several `TemporalBlock` layers and predicts from the final timestep.

`TCNWithAttention` uses the same TCN backbone, then applies multi-head self-attention over the TCN output sequence before prediction.

### `tcn_baseline.py`

This script is for testing the baseline alone.

It:

1. Loads participant data from the full `data/` folder.
2. Builds user-stratified folds.
3. Trains `TCNWithoutAttention`.
4. Evaluates validation loss.
5. Benchmarks inference latency.
6. Saves truth-vs-TCN prediction plots.
7. Saves a JSON result file.

### `compare_tcn_attention.py`

This script is for testing whether attention helps.

It:

1. Uses the same participant folds as the baseline.
2. Trains `TCNWithoutAttention`.
3. Trains `TCNWithAttention`.
4. Evaluates both on the same held-out users.
5. Benchmarks both models.
6. Saves plots with truth, TCN, and TCN + attention.
7. Saves a JSON result file comparing accuracy and latency.

## 5. Understanding Sequence Stride

The dataset creates overlapping windows of 100 timesteps. With `--sequence-stride 1`, every possible window is used. This is the full experiment, but it is very expensive on CPU.

Use larger strides for testing:

- `--sequence-stride 500`: Small Test
- `--sequence-stride 100`: Large Test
- `--sequence-stride 1`: Full Test

## 6. Reading Results

Baseline TCN result files contain:

- `summary.avg_loss`
- `summary.avg_latency_ms_per_sample`
- `summary.avg_throughput_samples_per_sec`
- `fold_results[*].prediction_plot`

Attention comparison result files contain:

- `summary.avg_loss_without`
- `summary.avg_loss_with`
- `summary.avg_improvement_percent`
- `summary.avg_latency_without_ms_per_sample`
- `summary.avg_latency_with_ms_per_sample`
- `summary.avg_latency_overhead_percent`
- `fold_results[*].prediction_plot`

Lower validation loss is better. Lower latency is better. For the attention experiment, the main question is whether the red TCN + attention line follows the black truth line better than the blue TCN line enough to justify any extra latency.
