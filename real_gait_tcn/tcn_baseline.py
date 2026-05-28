import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from compare_tcn_attention import (
    DATA_PATH,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TCN_BASELINE_RESULTS_PATH,
    TARGET_COLUMN,
    benchmark_inference,
    get_device,
    get_participant_list,
    prepare_fold_datasets,
    train_model,
)
from model import TCNWithoutAttention


def save_tcn_prediction_plot(
    model,
    loader,
    fold_number,
    device=None,
    output_dir=TCN_BASELINE_RESULTS_PATH / "plots" / "default",
    max_samples=300,
    plot_width=18,
    plot_height=4,
):
    """Save truth vs TCN prediction plot for one validation fold."""
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping prediction plot: matplotlib is not installed.")
        return None

    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    truth_values = []
    prediction_values = []
    collected = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            if isinstance(output, tuple):
                output = output[0]

            truth_values.append(y_batch.detach().cpu().numpy().reshape(-1))
            prediction_values.append(output.detach().cpu().numpy().reshape(-1))

            collected += y_batch.shape[0]
            if collected >= max_samples:
                break

    truth_values = np.concatenate(truth_values)[:max_samples]
    prediction_values = np.concatenate(prediction_values)[:max_samples]
    time_axis = np.arange(len(truth_values))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f"fold_{fold_number:02d}_tcn_predictions.png"

    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(time_axis, truth_values, color="black", linewidth=2, label="Truth")
    plt.plot(time_axis, prediction_values, color="blue", linewidth=1.5, label="TCN")
    plt.xlabel("Validation sample index")
    plt.ylabel(TARGET_COLUMN)
    plt.title(f"Fold {fold_number}: TCN Hip Moment Prediction")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"  Prediction plot saved to: {plot_file}")
    return str(plot_file)


def run_tcn_validation(
    n_splits=5,
    sequence_length=100,
    num_users=None,
    epochs=2,
    batch_size=32,
    sequence_stride=500,
    results_file=TCN_BASELINE_RESULTS_PATH / "json" / "tcn_small_test.json",
    device_name="auto",
    latency_batches=20,
    latency_iterations=10,
    plot_samples=300,
    plot_dir=TCN_BASELINE_RESULTS_PATH / "plots" / "small_test",
    plot_width=18,
    plot_height=4,
):
    """Run user-stratified validation for TCN without attention only."""
    results_file = Path(results_file)
    plot_dir = Path(plot_dir)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_participants = get_participant_list()

    if num_users is not None:
        participants = all_participants[:num_users]
        print(f"Testing with {num_users}/{len(all_participants)} users")
    else:
        participants = all_participants
        print(f"Testing with all {len(all_participants)} users")

    print(f"\n{'=' * 70}")
    print(f"TCN-ONLY USER-STRATIFIED {n_splits}-FOLD VALIDATION")
    print(f"{'=' * 70}")
    print(f"Data path: {DATA_PATH}")
    print(f"Total participants: {len(participants)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Sequence stride: {sequence_stride}")
    print(f"Input features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")

    if n_splits > len(participants):
        raise ValueError(f"n_splits={n_splits} cannot exceed participants={len(participants)}")

    device = get_device(device_name)
    print(f"Device: {device}")
    print("Latency benchmark: enabled")
    print("Prediction plots: enabled\n")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    start_time = time.perf_counter()

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(participants)):
        fold_number = fold_idx + 1
        print(f"{'=' * 70}")
        print(f"FOLD {fold_number}/{n_splits}")
        print(f"{'=' * 70}")

        train_participants = [participants[i] for i in train_idx]
        val_participants = [participants[i] for i in val_idx]

        print(f"Training participants: {len(train_participants)} users")
        print(f"Validation participants: {len(val_participants)} users\n")

        train_dataset, val_dataset = prepare_fold_datasets(
            train_participants,
            val_participants,
            sequence_length,
            sequence_stride,
        )

        if train_dataset is None:
            print(f"Skipping fold {fold_number} (insufficient data)")
            continue

        print(f"\nTraining samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print("\n  Training TCN WITHOUT Attention...")
        model = TCNWithoutAttention(input_features=len(FEATURE_COLUMNS))
        val_loss = train_model(
            model,
            train_loader,
            val_loader,
            model_name="TCN",
            epochs=epochs,
            device=device,
        )
        print(f"  Best Val Loss (TCN): {val_loss:.6f}")

        latency = benchmark_inference(
            model,
            val_loader,
            model_name="TCN",
            device=device,
            max_batches=latency_batches,
            iterations=latency_iterations,
        )

        prediction_plot = save_tcn_prediction_plot(
            model,
            val_loader,
            fold_number=fold_number,
            device=device,
            output_dir=plot_dir,
            max_samples=plot_samples,
            plot_width=plot_width,
            plot_height=plot_height,
        )

        fold_results.append(
            {
                "fold": fold_number,
                "train_participants": train_participants,
                "val_participants": val_participants,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "val_loss": val_loss,
                "latency": latency,
                "prediction_plot": prediction_plot,
            }
        )

    elapsed_seconds = time.perf_counter() - start_time

    avg_loss = float(np.mean([r["val_loss"] for r in fold_results]))
    std_loss = float(np.std([r["val_loss"] for r in fold_results]))
    avg_latency = float(np.mean([r["latency"]["per_sample_latency_ms"] for r in fold_results]))
    avg_throughput = float(np.mean([r["latency"]["throughput_samples_per_sec"] for r in fold_results]))

    print(f"\n{'=' * 70}")
    print("TCN-ONLY SUMMARY")
    print(f"{'=' * 70}")
    print(f"Average validation loss: {avg_loss:.6f} +/- {std_loss:.6f}")
    print(f"Average latency:         {avg_latency:.6f} ms/sample")
    print(f"Average throughput:      {avg_throughput:.1f} samples/sec")
    print(f"Elapsed time:            {elapsed_seconds / 60:.2f} minutes")

    output = {
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
        "elapsed_seconds": elapsed_seconds,
        "fold_results": fold_results,
        "summary": {
            "avg_loss": avg_loss,
            "std_loss": std_loss,
            "avg_latency_ms_per_sample": avg_latency,
            "avg_throughput_samples_per_sec": avg_throughput,
        },
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCN-only user-stratified validation.")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=500)
    parser.add_argument("--num-users", type=int, default=0, help="Use all users with --num-users 0.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--results-file", default=str(TCN_BASELINE_RESULTS_PATH / "json" / "tcn_small_test.json"))
    parser.add_argument("--device", default="auto", help="Use auto, cpu, cuda, or mps.")
    parser.add_argument("--latency-batches", type=int, default=20, help="Validation batches to time.")
    parser.add_argument("--latency-iterations", type=int, default=10, help="Timed forward passes per latency batch.")
    parser.add_argument("--plot-samples", type=int, default=300, help="Validation samples to draw per fold.")
    parser.add_argument("--plot-dir", default=str(TCN_BASELINE_RESULTS_PATH / "plots" / "small_test"), help="Directory for saved prediction plots.")
    parser.add_argument("--plot-width", type=float, default=18, help="Prediction plot width in inches.")
    parser.add_argument("--plot-height", type=float, default=4, help="Prediction plot height in inches.")
    args = parser.parse_args()

    run_tcn_validation(
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
