import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import argparse
import json
import time
import os

from model import TCNWithAttention, TCNWithoutAttention


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "doi_10_5061_dryad_8kprr4xsv__v20240321"
RESULTS_PATH = PROJECT_ROOT / "results"
ATTENTION_RESULTS_PATH = RESULTS_PATH / "attention_comparison"
TCN_BASELINE_RESULTS_PATH = RESULTS_PATH / "tcn_baseline"

FEATURE_COLUMNS = [
    "enc_angle_l",
    "enc_velo_l",
    "thigh_accel_x_l",
    "thigh_accel_y_l",
    "thigh_accel_z_l",
    "thigh_gyro_x_l",
    "thigh_gyro_y_l",
    "thigh_gyro_z_l",
    "pelvis_accel_x",
    "pelvis_accel_y",
    "pelvis_accel_z",
    "pelvis_gyro_x",
    "pelvis_gyro_y",
    "pelvis_gyro_z",
]

TARGET_COLUMN = "hip_moment_l"


def get_device(device_name="auto"):
    """Pick a training/benchmark device."""
    if device_name != "auto":
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize_device(device):
    """Synchronize accelerator timing when needed."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


class ParticipantSequenceDataset(Dataset):
    """Lazy window dataset so cross-validation does not materialize every window."""

    def __init__(self, participant_arrays, sequence_length=100, sequence_stride=1):
        self.participant_arrays = participant_arrays
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.cumulative_lengths = []

        total = 0
        for X, _ in self.participant_arrays:
            num_windows = max(0, (len(X) - sequence_length + sequence_stride - 1) // sequence_stride)
            total += num_windows
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        participant_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
        prev_total = 0 if participant_idx == 0 else self.cumulative_lengths[participant_idx - 1]
        local_idx = idx - prev_total
        start = local_idx * self.sequence_stride

        X, y = self.participant_arrays[participant_idx]
        x_window = X[start:start + self.sequence_length]
        y_value = y[start + self.sequence_length]

        return x_window, y_value.unsqueeze(0)


def get_participant_list():
    """Get list of all participants (AB01-AB34)"""
    participants = []
    for i in range(1, 35):
        participants.append(f"AB{i:02d}")
    return participants


def load_participant_data(participant_id, sequence_length=100):
    """Load and merge all trials for a participant"""
    participant_path = DATA_PATH / participant_id
    
    if not participant_path.exists():
        return None
    
    all_data = []
    
    # Iterate through all trials for this participant
    for trial_dir in sorted(participant_path.iterdir()):
        if not trial_dir.is_dir():
            continue
        
        try:
            exo_file = trial_dir / "exo.csv"
            moment_file = trial_dir / "moment.csv"
            
            if not exo_file.exists() or not moment_file.exists():
                continue
            
            exo = pd.read_csv(exo_file)
            moment = pd.read_csv(moment_file)
            
            # Merge on time
            df = pd.merge(exo, moment, on="time", how="inner")
            
            # Select features and target
            df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
            
            if len(df) < sequence_length:
                continue
            
            all_data.append(df)
        
        except Exception as e:
            print(f"Error loading {trial_dir}: {e}")
            continue
    
    if not all_data:
        return None
    
    # Concatenate all trials
    combined_data = pd.concat(all_data, ignore_index=True)
    
    if len(combined_data) < sequence_length:
        return None
    
    return combined_data


def load_participant_arrays(participant_id, sequence_length=100, sequence_stride=1):
    """Load one participant as tensors for lazy sequence slicing."""
    data = load_participant_data(participant_id, sequence_length)
    if data is None:
        return None

    X = data[FEATURE_COLUMNS].values
    y = data[TARGET_COLUMN].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    num_windows = max(0, (len(X) - sequence_length + sequence_stride - 1) // sequence_stride)
    if num_windows == 0:
        return None

    return X, y, num_windows


def create_sequences(X, y, sequence_length=100):
    """Create overlapping sequences from data"""
    X_windows = []
    y_values = []
    
    for i in range(len(X) - sequence_length):
        X_windows.append(X[i:i + sequence_length])
        y_values.append(y[i + sequence_length])
    
    if not X_windows:
        return None, None
    
    X_windows = torch.tensor(np.array(X_windows), dtype=torch.float32)
    y_values = torch.tensor(np.array(y_values), dtype=torch.float32).unsqueeze(1)
    
    return X_windows, y_values


def prepare_fold_data(train_participants, val_participants, sequence_length=100):
    """Prepare data for training and validation folds"""
    train_X_list = []
    train_y_list = []
    val_X_list = []
    val_y_list = []
    
    print(f"  Loading train participants: {train_participants}")
    for participant in train_participants:
        data = load_participant_data(participant, sequence_length)
        if data is None:
            print(f"    ⚠️ Skipped {participant} (insufficient data)")
            continue
        
        X = data[FEATURE_COLUMNS].values
        y = data[TARGET_COLUMN].values
        
        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_seq, y_seq = create_sequences(X, y, sequence_length)
        if X_seq is not None:
            train_X_list.append(X_seq)
            train_y_list.append(y_seq)
            print(f"    ✓ {participant}: {len(X_seq)} samples")
    
    print(f"  Loading val participants: {val_participants}")
    for participant in val_participants:
        data = load_participant_data(participant, sequence_length)
        if data is None:
            print(f"    ⚠️ Skipped {participant} (insufficient data)")
            continue
        
        X = data[FEATURE_COLUMNS].values
        y = data[TARGET_COLUMN].values
        
        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_seq, y_seq = create_sequences(X, y, sequence_length)
        if X_seq is not None:
            val_X_list.append(X_seq)
            val_y_list.append(y_seq)
            print(f"    ✓ {participant}: {len(X_seq)} samples")
    
    if not train_X_list or not val_X_list:
        return None, None, None, None
    
    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)
    val_X = torch.cat(val_X_list, dim=0)
    val_y = torch.cat(val_y_list, dim=0)
    
    return train_X, train_y, val_X, val_y


def prepare_fold_datasets(train_participants, val_participants, sequence_length=100, sequence_stride=1):
    """Prepare lazy datasets for training and validation folds."""
    train_arrays = []
    val_arrays = []

    print(f"  Loading train participants: {train_participants}")
    for participant in train_participants:
        participant_data = load_participant_arrays(participant, sequence_length, sequence_stride)
        if participant_data is None:
            print(f"    Skipped {participant} (insufficient data)")
            continue

        X, y, num_windows = participant_data
        train_arrays.append((X, y))
        print(f"    OK {participant}: {num_windows} samples")

    print(f"  Loading val participants: {val_participants}")
    for participant in val_participants:
        participant_data = load_participant_arrays(participant, sequence_length, sequence_stride)
        if participant_data is None:
            print(f"    Skipped {participant} (insufficient data)")
            continue

        X, y, num_windows = participant_data
        val_arrays.append((X, y))
        print(f"    OK {participant}: {num_windows} samples")

    if not train_arrays or not val_arrays:
        return None, None

    train_dataset = ParticipantSequenceDataset(train_arrays, sequence_length, sequence_stride)
    val_dataset = ParticipantSequenceDataset(val_arrays, sequence_length, sequence_stride)

    return train_dataset, val_dataset


def train_model(model, train_loader, val_loader, model_name="", epochs=20, device=None):
    """Train a model"""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            
            output = model(x_batch)
            if isinstance(output, tuple):
                prediction = output[0]
            else:
                prediction = output
            
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(x_batch)
                if isinstance(output, tuple):
                    prediction = output[0]
                else:
                    prediction = output
                
                loss = criterion(prediction, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Show progress every epoch
        print(f"      [{model_name}] Epoch {epoch+1:02d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
    return best_val_loss


def benchmark_inference(model, loader, model_name="", device=None, max_batches=20, iterations=10):
    """Benchmark model inference on a bounded number of validation batches."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    latencies = []
    total_samples = 0
    measured_batches = 0

    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x_batch = x_batch.to(device)

            _ = model(x_batch)
            synchronize_device(device)

            for _ in range(iterations):
                start_time = time.perf_counter()
                _ = model(x_batch)
                synchronize_device(device)
                end_time = time.perf_counter()

                latencies.append((end_time - start_time) * 1000)
                total_samples += x_batch.shape[0]

            measured_batches += 1

    if not latencies:
        return None

    latencies = np.array(latencies, dtype=np.float64)
    avg_latency_ms = float(latencies.mean())
    std_latency_ms = float(latencies.std())
    min_latency_ms = float(latencies.min())
    max_latency_ms = float(latencies.max())
    per_sample_latency_ms = float(latencies.sum() / total_samples)
    throughput_samples_per_sec = float(total_samples * 1000 / latencies.sum())

    results = {
        "model_name": model_name,
        "device": str(device),
        "max_batches": max_batches,
        "iterations_per_batch": iterations,
        "measured_batches": measured_batches,
        "total_timed_samples": total_samples,
        "avg_batch_latency_ms": avg_latency_ms,
        "std_batch_latency_ms": std_latency_ms,
        "min_batch_latency_ms": min_latency_ms,
        "max_batch_latency_ms": max_latency_ms,
        "per_sample_latency_ms": per_sample_latency_ms,
        "throughput_samples_per_sec": throughput_samples_per_sec,
    }

    print(
        f"  Latency ({model_name}, {device}): "
        f"{per_sample_latency_ms:.6f} ms/sample, "
        f"{throughput_samples_per_sec:.1f} samples/sec"
    )

    return results


def save_prediction_plot(
    model_without,
    model_with,
    loader,
    fold_number,
    device=None,
    output_dir=ATTENTION_RESULTS_PATH / "plots" / "default",
    max_samples=300,
    plot_width=18,
    plot_height=4,
):
    """Save ground-truth vs prediction plot for one validation fold."""
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping prediction plot: matplotlib is not installed.")
        return None

    if device is None:
        device = torch.device("cpu")

    model_without = model_without.to(device)
    model_with = model_with.to(device)
    model_without.eval()
    model_with.eval()

    truth_values = []
    without_values = []
    with_values = []
    collected = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            output_without = model_without(x_batch)
            output_with = model_with(x_batch)

            if isinstance(output_without, tuple):
                output_without = output_without[0]
            if isinstance(output_with, tuple):
                output_with = output_with[0]

            truth_values.append(y_batch.detach().cpu().numpy().reshape(-1))
            without_values.append(output_without.detach().cpu().numpy().reshape(-1))
            with_values.append(output_with.detach().cpu().numpy().reshape(-1))

            collected += y_batch.shape[0]
            if collected >= max_samples:
                break

    truth_values = np.concatenate(truth_values)[:max_samples]
    without_values = np.concatenate(without_values)[:max_samples]
    with_values = np.concatenate(with_values)[:max_samples]
    time_axis = np.arange(len(truth_values))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f"fold_{fold_number:02d}_moment_predictions.png"

    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(time_axis, truth_values, color="black", linewidth=2, label="Truth")
    plt.plot(time_axis, without_values, color="blue", linewidth=1.5, label="TCN")
    plt.plot(time_axis, with_values, color="red", linewidth=1.5, label="TCN + Attention")
    plt.xlabel("Validation sample index")
    plt.ylabel(TARGET_COLUMN)
    plt.title(f"Fold {fold_number}: Hip Moment Prediction Comparison")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"  Prediction plot saved to: {plot_file}")
    return str(plot_file)


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    criterion = nn.MSELoss()
    
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            if isinstance(output, tuple):
                prediction = output[0]
            else:
                prediction = output
            
            loss = criterion(prediction, y_batch)
            test_loss += loss.item()
    
    return test_loss / len(test_loader)


def run_cross_validation(
    n_splits=5,
    sequence_length=100,
    num_users=None,
    epochs=2,
    batch_size=32,
    sequence_stride=500,
    results_file=ATTENTION_RESULTS_PATH / "json" / "cv_small_test.json",
    device_name="auto",
    latency_batches=20,
    latency_iterations=10,
    plot_samples=300,
    plot_dir=ATTENTION_RESULTS_PATH / "plots" / "small_test",
    plot_width=18,
    plot_height=4,
):
    """Run user-stratified 5-fold cross-validation
    
    Args:
        n_splits: Number of folds
        sequence_length: Length of temporal sequences
        num_users: Number of users to test with (None = all 34 users)
    """
    
    results_file = Path(results_file)
    plot_dir = Path(plot_dir)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_participants = get_participant_list()
    
    # Use subset if specified
    if num_users is not None:
        participants = all_participants[:num_users]
        print(f"Testing with {num_users}/{len(all_participants)} users")
    else:
        participants = all_participants
        print(f"Testing with all {len(all_participants)} users")
    
    print(f"\n{'='*70}")
    print(f"USER-STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Total participants: {len(participants)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Sequence stride: {sequence_stride}")
    print(f"Input features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}\n")

    device = get_device(device_name)
    print(f"Device: {device}")
    print(
        f"Latency benchmark: enabled "
        f"({latency_batches} batches, {latency_iterations} iterations/batch)\n"
    )
    print(
        f"Prediction plots: enabled "
        f"({plot_samples} samples/fold, {plot_width}x{plot_height} in, dir={plot_dir})\n"
    )

    if n_splits > len(participants):
        raise ValueError(f"n_splits={n_splits} cannot exceed participants={len(participants)}")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(participants)):
        print(f"{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*70}")
        
        train_participants = [participants[i] for i in train_idx]
        val_participants = [participants[i] for i in val_idx]
        
        print(f"Training participants: {len(train_participants)} users")
        print(f"Validation participants: {len(val_participants)} users\n")
        
        # Prepare data for fold
        train_dataset, val_dataset = prepare_fold_datasets(
            train_participants,
            val_participants,
            sequence_length,
            sequence_stride
        )
        
        if train_dataset is None:
            print(f"Skipping fold {fold_idx + 1} (insufficient data)")
            continue
        
        print(f"\nTraining samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train TCN WITHOUT Attention
        print(f"\n  Training TCN WITHOUT Attention...")
        model_without = TCNWithoutAttention(input_features=len(FEATURE_COLUMNS))
        val_loss_without = train_model(model_without, train_loader, val_loader, 
                                       model_name="WITHOUT", epochs=epochs, device=device)
        print(f"  Best Val Loss (WITHOUT): {val_loss_without:.6f}")

        latency_without = benchmark_inference(
            model_without,
            val_loader,
            model_name="WITHOUT",
            device=device,
            max_batches=latency_batches,
            iterations=latency_iterations,
        )
        
        # Train TCN WITH Attention
        print(f"\n  Training TCN WITH Attention...")
        model_with = TCNWithAttention(input_features=len(FEATURE_COLUMNS))
        val_loss_with = train_model(model_with, train_loader, val_loader, 
                                    model_name="WITH", epochs=epochs, device=device)
        print(f"  Best Val Loss (WITH): {val_loss_with:.6f}")

        latency_with = benchmark_inference(
            model_with,
            val_loader,
            model_name="WITH",
            device=device,
            max_batches=latency_batches,
            iterations=latency_iterations,
        )

        latency_overhead = (
            (
                latency_with["per_sample_latency_ms"]
                - latency_without["per_sample_latency_ms"]
            )
            / latency_without["per_sample_latency_ms"]
            * 100
        )
        throughput_reduction = (
            (
                latency_without["throughput_samples_per_sec"]
                - latency_with["throughput_samples_per_sec"]
            )
            / latency_without["throughput_samples_per_sec"]
            * 100
        )

        prediction_plot = save_prediction_plot(
            model_without,
            model_with,
            val_loader,
            fold_number=fold_idx + 1,
            device=device,
            output_dir=plot_dir,
            max_samples=plot_samples,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        
        # Store results
        fold_result = {
            "fold": fold_idx + 1,
            "train_participants": train_participants,
            "val_participants": val_participants,
            "val_loss_without": val_loss_without,
            "val_loss_with": val_loss_with,
            "improvement": (val_loss_without - val_loss_with) / val_loss_without * 100,
            "latency_without": latency_without,
            "latency_with": latency_with,
            "latency_overhead_percent": latency_overhead,
            "throughput_reduction_percent": throughput_reduction,
            "prediction_plot": prediction_plot,
        }
        fold_results.append(fold_result)
        
        print(f"\n  Improvement: {fold_result['improvement']:+.2f}%")
        print(f"  Attention latency overhead: {latency_overhead:+.2f}%")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Fold':<8} {'WITHOUT (Val Loss)':<20} {'WITH (Val Loss)':<20} {'Improvement':<15}")
    print("-" * 70)
    
    for result in fold_results:
        print(
            f"{result['fold']:<8} "
            f"{result['val_loss_without']:<20.6f} "
            f"{result['val_loss_with']:<20.6f} "
            f"{result['improvement']:>+14.2f}%"
        )
    
    # Calculate averages
    avg_without = np.mean([r["val_loss_without"] for r in fold_results])
    avg_with = np.mean([r["val_loss_with"] for r in fold_results])
    avg_improvement = np.mean([r["improvement"] for r in fold_results])
    std_without = np.std([r["val_loss_without"] for r in fold_results])
    std_with = np.std([r["val_loss_with"] for r in fold_results])
    std_improvement = np.std([r["improvement"] for r in fold_results])

    avg_latency_without = np.mean([r["latency_without"]["per_sample_latency_ms"] for r in fold_results])
    avg_latency_with = np.mean([r["latency_with"]["per_sample_latency_ms"] for r in fold_results])
    avg_latency_overhead = np.mean([r["latency_overhead_percent"] for r in fold_results])
    avg_throughput_without = np.mean([r["latency_without"]["throughput_samples_per_sec"] for r in fold_results])
    avg_throughput_with = np.mean([r["latency_with"]["throughput_samples_per_sec"] for r in fold_results])
    avg_throughput_reduction = np.mean([r["throughput_reduction_percent"] for r in fold_results])
    
    print("-" * 70)
    print(
        f"{'AVG':<8} "
        f"{avg_without:<20.6f} "
        f"{avg_with:<20.6f} "
        f"{avg_improvement:>+14.2f}%"
    )
    print(
        f"{'STD':<8} "
        f"{std_without:<20.6f} "
        f"{std_with:<20.6f} "
        f"{std_improvement:>14.2f}%"
    )
    
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    print(f"\nAttention Mechanism Impact on User Generalization:")
    print(f"  Average validation loss WITHOUT attention: {avg_without:.6f} ± {std_without:.6f}")
    print(f"  Average validation loss WITH attention:    {avg_with:.6f} ± {std_with:.6f}")
    print(f"  Average improvement:                       {avg_improvement:+.2f}% ± {std_improvement:.2f}%")

    print(f"\nInference Latency Impact ({device}):")
    print(f"  Average latency WITHOUT attention:         {avg_latency_without:.6f} ms/sample")
    print(f"  Average latency WITH attention:            {avg_latency_with:.6f} ms/sample")
    print(f"  Average attention latency overhead:        {avg_latency_overhead:+.2f}%")
    print(f"  Average throughput WITHOUT attention:      {avg_throughput_without:.1f} samples/sec")
    print(f"  Average throughput WITH attention:         {avg_throughput_with:.1f} samples/sec")
    print(f"  Average throughput reduction:              {avg_throughput_reduction:+.2f}%")
    
    if avg_improvement > 0:
        print(f"\nAttention IMPROVES generalization to unseen users")
        print(f"  The model with attention adapts better to different user biomechanics")
    else:
        print(f"\nAttention does NOT improve generalization")
    
    # Save results
    with open(results_file, "w") as f:
        json.dump(
            {
                "n_splits": n_splits,
                "sequence_length": sequence_length,
                "sequence_stride": sequence_stride,
                "num_features": len(FEATURE_COLUMNS),
                "epochs": epochs,
                "batch_size": batch_size,
                "device": str(device),
                "latency_batches": latency_batches,
                "latency_iterations": latency_iterations,
                "plot_samples": plot_samples,
                "plot_dir": str(plot_dir),
                "plot_width": plot_width,
                "plot_height": plot_height,
                "fold_results": fold_results,
                "summary": {
                    "avg_loss_without": float(avg_without),
                    "avg_loss_with": float(avg_with),
                    "std_loss_without": float(std_without),
                    "std_loss_with": float(std_with),
                    "avg_improvement_percent": float(avg_improvement),
                    "std_improvement_percent": float(std_improvement),
                    "avg_latency_without_ms_per_sample": float(avg_latency_without),
                    "avg_latency_with_ms_per_sample": float(avg_latency_with),
                    "avg_latency_overhead_percent": float(avg_latency_overhead),
                    "avg_throughput_without_samples_per_sec": float(avg_throughput_without),
                    "avg_throughput_with_samples_per_sec": float(avg_throughput_with),
                    "avg_throughput_reduction_percent": float(avg_throughput_reduction),
                }
            },
            f,
            indent=2
        )
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User-stratified cross-validation for real gait TCN models.")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=500)
    parser.add_argument("--num-users", type=int, default=0, help="Use all users with --num-users 0.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--results-file", default=str(ATTENTION_RESULTS_PATH / "json" / "cv_small_test.json"))
    parser.add_argument("--device", default="auto", help="Use auto, cpu, cuda, or mps.")
    parser.add_argument("--latency-batches", type=int, default=20, help="Validation batches to time per model.")
    parser.add_argument("--latency-iterations", type=int, default=10, help="Timed forward passes per latency batch.")
    parser.add_argument("--plot-samples", type=int, default=300, help="Validation samples to draw per fold.")
    parser.add_argument("--plot-dir", default=str(ATTENTION_RESULTS_PATH / "plots" / "small_test"), help="Directory for saved prediction plots.")
    parser.add_argument("--plot-width", type=float, default=18, help="Prediction plot width in inches.")
    parser.add_argument("--plot-height", type=float, default=4, help="Prediction plot height in inches.")
    args = parser.parse_args()

    run_cross_validation(
        n_splits=args.splits,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        num_users=None if args.num_users == 0 else args.num_users,
        epochs=args.epochs,
        batch_size=args.batch_size,
        results_file=args.results_file,
        device_name=args.device,
        latency_batches=args.latency_batches,
        latency_iterations=args.latency_iterations,
        plot_samples=args.plot_samples,
        plot_dir=args.plot_dir,
        plot_width=args.plot_width,
        plot_height=args.plot_height,
    )
