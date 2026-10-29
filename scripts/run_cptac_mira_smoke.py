#!/usr/bin/env python3
"""Prepare and run the CPTAC LUAD/LSCC MIRA-seg smoke test."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATASET = Path("/mnt/e/data_mira/dataset/CPTAC-LUAD-v13")
RESULTS = Path("/home/jonathan/data/CPTAC_STAMP")
CHECKPOINT = Path("/home/jonathan/Projects/code/STAMP/checkpoint/last.ckpt")
IGNITE_CHECKPOINT = Path(
    "/home/jonathan/Projects/code/STAMP/checkpoint/lung-ignite-best.ckpt"
)
NEXTGEN_CHECKPOINT = Path(
    "/home/jonathan/Projects/code/STAMP/checkpoint/lung-nextgen-best.ckpt"
)
VICREG_CHECKPOINT = Path(
    "/home/jonathan/Projects/code/mira-ml-template/runs/"
    "vicreg-mit-b2-20x-20261026-204825-1183d69c/checkpoints/last.ckpt"
)
FEATURE_ROOT = RESULTS / "features_20x"
FEATURE_DIR = FEATURE_ROOT / "mira_seg-bb0471e2-1024px-bf16"
IGNITE_FEATURE_ROOT = RESULTS / "features_20x_lung_ignite"
NEXTGEN_FEATURE_ROOT = RESULTS / "features_20x_lung_nextgen"
NEXTGEN_10X_FEATURE_ROOT = RESULTS / "features_10x_lung_nextgen"
IMAGENET_FEATURE_ROOT = RESULTS / "features_20x_imagenet"
VICREG_FEATURE_ROOT = RESULTS / "features_20x_vicreg"
AUTOENCODER_FEATURE_ROOT = RESULTS / "features_20x_autoencoder"
IGNITE_FEATURE_DIR = IGNITE_FEATURE_ROOT / "mira_seg-c64e638e-1024px-bf16"
NEXTGEN_FEATURE_DIR = NEXTGEN_FEATURE_ROOT / "mira_seg-489c6bcf-512px-bf16"
IMAGENET_FEATURE_DIR = (
    IMAGENET_FEATURE_ROOT / "mira_imagenet-mit_b2-imagenet-1024px-bf16"
)
VICREG_FEATURE_DIR = (
    VICREG_FEATURE_ROOT / "vicreg-10a53e2d-224px-0.5mpp-bf16"
)
AUTOENCODER_FEATURE_DIR = (
    AUTOENCODER_FEATURE_ROOT
    / "mira_autoencoder-b15b62f6-512px-0.5mpp-bf16"
)
CONFIG_PATH = RESULTS / "config.yaml"
CV5_CONFIG_PATH = RESULTS / "config_crossval_5fold.yaml"
IGNITE_CONFIG_PATH = RESULTS / "config_lung_ignite.yaml"
NEXTGEN_CONFIG_PATH = RESULTS / "config_lung_nextgen.yaml"
NEXTGEN_10X_CONFIG_PATH = RESULTS / "config_lung_nextgen_10x.yaml"
IMAGENET_CONFIG_PATH = RESULTS / "config_imagenet.yaml"
VICREG_CONFIG_PATH = RESULTS / "config_vicreg.yaml"
AUTOENCODER_CONFIG_PATH = RESULTS / "config_autoencoder.yaml"
MIL_COMPARISON_ROOT = RESULTS / "mil_comparison_cv5"
MIL_COMPARISON_CONFIGS = {
    "mira": RESULTS / "config_cv5_mira.yaml",
    "ignite": RESULTS / "config_cv5_ignite.yaml",
    "nextgen": RESULTS / "config_cv5_nextgen.yaml",
    "imagenet": RESULTS / "config_cv5_imagenet.yaml",
    "vicreg": RESULTS / "config_cv5_vicreg.yaml",
    "autoencoder": RESULTS / "config_cv5_autoencoder.yaml",
}
MIL_TEST_ROOT = RESULTS / "mil_test_independent"
MIL_TEST_CONFIGS = {
    name: RESULTS / f"config_test_{name}.yaml" for name in MIL_COMPARISON_CONFIGS
}
STATUS_PATH = RESULTS / "status.json"
PIPELINE_LOG = RESULTS / "pipeline.log"


def write_status(stage: str, state: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def prepare() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATASET / "all_slides_metadata.csv")
    metadata["PATIENT"] = metadata["Case_ID"].fillna(metadata["inferred_case_id"])
    metadata["DIAGNOSIS"] = metadata["local_cohort"]
    metadata = metadata[
        metadata["DIAGNOSIS"].isin(["LUAD", "LSCC"]) & metadata["PATIENT"].notna()
    ].copy()
    metadata = metadata[metadata["img_file"].map(lambda p: Path(p).is_file())]

    patient_labels = metadata[["PATIENT", "DIAGNOSIS"]].drop_duplicates()
    if patient_labels["PATIENT"].duplicated().any():
        raise RuntimeError("At least one patient occurs in both diagnostic classes")

    # An odd cohort cannot be perfectly balanced: use the closest possible
    # allocation, with one additional LUAD patient.
    patient_labels = pd.concat(
        [
            patient_labels[patient_labels["DIAGNOSIS"] == "LUAD"].sample(
                n=53, random_state=42
            ),
            patient_labels[patient_labels["DIAGNOSIS"] == "LSCC"].sample(
                n=52, random_state=42
            ),
        ],
        ignore_index=True,
    )

    train_patients, remaining = train_test_split(
        patient_labels,
        train_size=75,
        test_size=30,
        stratify=patient_labels["DIAGNOSIS"],
        random_state=42,
    )
    valid_patients, test_patients = train_test_split(
        remaining,
        train_size=15,
        test_size=15,
        stratify=remaining["DIAGNOSIS"],
        random_state=42,
    )

    split_frames = []
    for split, frame in (
        ("train", train_patients),
        ("validation", valid_patients),
        ("test", test_patients),
    ):
        part = frame.copy()
        part["SPLIT"] = split
        split_frames.append(part)
    cohort = pd.concat(split_frames, ignore_index=True)
    cohort.to_csv(RESULTS / "patient_cohort.csv", index=False)

    selected = metadata.merge(cohort, on=["PATIENT", "DIAGNOSIS"], how="inner")
    selected["WSI"] = selected["img_file"].map(
        lambda p: Path(p).relative_to(DATASET).as_posix()
    )
    selected["FILENAME"] = selected["WSI"].map(
        lambda p: str(Path(p).with_suffix(".h5"))
    )
    selected[["WSI"]].to_csv(RESULTS / "wsi_list.csv", index=False, header=False)
    selected[
        ["PATIENT", "DIAGNOSIS", "SPLIT", "WSI", "FILENAME", "Specimen_Type"]
    ].to_csv(RESULTS / "slide_manifest.csv", index=False)

    training_clini = cohort[cohort["SPLIT"].isin(["train", "validation"])][
        ["PATIENT", "DIAGNOSIS"]
    ]
    training_slides = selected[selected["SPLIT"].isin(["train", "validation"])][
        ["PATIENT", "FILENAME"]
    ]
    test_clini = cohort[cohort["SPLIT"] == "test"][["PATIENT", "DIAGNOSIS"]]
    test_slides = selected[selected["SPLIT"] == "test"][["PATIENT", "FILENAME"]]
    training_clini.to_csv(RESULTS / "training_clini.csv", index=False)
    training_slides.to_csv(RESULTS / "training_slides.csv", index=False)
    test_clini.to_csv(RESULTS / "test_clini.csv", index=False)
    test_slides.to_csv(RESULTS / "test_slides.csv", index=False)

    training_dir = RESULTS / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "splits": [
            {
                "train_patients": sorted(train_patients["PATIENT"].tolist()),
                "test_patients": sorted(valid_patients["PATIENT"].tolist()),
            }
        ]
    }
    (training_dir / "splits.json").write_text(
        json.dumps(split_payload, indent=2) + "\n"
    )

    config = {
        "preprocessing": {
            "output_dir": str(FEATURE_ROOT),
            "wsi_dir": str(DATASET),
            "wsi_list": str(RESULTS / "wsi_list.csv"),
            "extractor": "mira_seg",
            "mira_seg_checkpoint_path": str(CHECKPOINT),
            "device": "cuda:0",
            "max_workers": 8,
            "generate_hash": False,
            # Native 20x inference: 1024 px at 0.5 um/px covers 512 um.
            "tile_size_px": 1024,
            "tile_size_um": 512.0,
        },
        "crossval": {
            "output_dir": str(training_dir),
            "clini_table": str(RESULTS / "training_clini.csv"),
            "slide_table": str(RESULTS / "training_slides.csv"),
            "feature_dir": str(FEATURE_DIR),
            "task": "classification",
            "ground_truth_label": "DIAGNOSIS",
            "categories": ["LSCC", "LUAD"],
            "patient_label": "PATIENT",
            "filename_label": "FILENAME",
            "n_splits": 2,
        },
        "deployment": {
            "output_dir": str(RESULTS / "test"),
            "checkpoint_paths": [str(training_dir / "split-0" / "model.ckpt")],
            "clini_table": str(RESULTS / "test_clini.csv"),
            "slide_table": str(RESULTS / "test_slides.csv"),
            "feature_dir": str(FEATURE_DIR),
            "ground_truth_label": "DIAGNOSIS",
            "patient_label": "PATIENT",
            "filename_label": "FILENAME",
            "num_workers": 8,
            "accelerator": "gpu",
        },
        "statistics": {
            "output_dir": str(RESULTS / "test" / "statistics"),
            "task": "classification",
            "ground_truth_label": "DIAGNOSIS",
            "true_class": "LUAD",
            "pred_csvs": [str(RESULTS / "test" / "patient-preds.csv")],
        },
        "advanced_config": {
            "bag_size": 512,
            "num_workers": 8,
            "batch_size": 16,
            "max_epochs": 8,
            "patience": 3,
            "accelerator": "gpu",
            "max_lr": 0.0001,
            "div_factor": 25.0,
            "model_name": "vit",
            "seed": 42,
            "model_params": {
                "vit": {
                    "dim_model": 512,
                    "dim_feedforward": 512,
                    "n_heads": 8,
                    "n_layers": 2,
                    "dropout": 0.0,
                    "use_alibi": False,
                },
                "trans_mil": {"dim_hidden": 512},
                "mlp": {"dim_hidden": 512, "num_layers": 2, "dropout": 0.25},
                "linear": {
                    "num_encoder_heads": 8,
                    "num_decoder_heads": 8,
                    "num_encoder_layers": 2,
                    "num_decoder_layers": 2,
                    "dim_feedforward": 2048,
                    "positional_encoding": True,
                    "learning_rate": 0.0001,
                },
                "barspoon": {
                    "d_model": 512,
                    "num_encoder_heads": 8,
                    "num_decoder_heads": 8,
                    "num_encoder_layers": 2,
                    "num_decoder_layers": 2,
                    "dim_feedforward": 2048,
                    "positional_encoding": True,
                    "learning_rate": 0.0001,
                },
            },
        },
    }
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False))

    counts = cohort.groupby(["SPLIT", "DIAGNOSIS"]).size().unstack(fill_value=0)
    write_status(
        "prepared",
        "complete",
        patients=int(len(cohort)),
        slides=int(len(selected)),
        split_counts={
            split: {label: int(value) for label, value in row.items()}
            for split, row in counts.iterrows()
        },
    )


def prepare_cv5_config() -> None:
    """Create a five-fold development-cohort config without touching the holdout."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["crossval"]["output_dir"] = str(RESULTS / "training_cv5")
    config["crossval"]["n_splits"] = 5
    CV5_CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False))


def prepare_ignite_config() -> None:
    """Create a preprocessing-only config for the lung IGNITE checkpoint."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preprocessing = config["preprocessing"]
    preprocessing["output_dir"] = str(IGNITE_FEATURE_ROOT)
    preprocessing["mira_seg_checkpoint_path"] = str(IGNITE_CHECKPOINT)
    IGNITE_CONFIG_PATH.write_text(
        yaml.safe_dump({"preprocessing": preprocessing}, sort_keys=False)
    )


def prepare_nextgen_config() -> None:
    """Create a preprocessing-only config for the lung NextGen checkpoint."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preprocessing = config["preprocessing"]
    preprocessing["output_dir"] = str(NEXTGEN_FEATURE_ROOT)
    preprocessing["mira_seg_checkpoint_path"] = str(NEXTGEN_CHECKPOINT)
    # Native 20x inference for this checkpoint: 512 px at 0.5 um/px.
    preprocessing["tile_size_px"] = 512
    preprocessing["tile_size_um"] = 256.0
    NEXTGEN_CONFIG_PATH.write_text(
        yaml.safe_dump({"preprocessing": preprocessing}, sort_keys=False)
    )


def prepare_nextgen_10x_config() -> None:
    """Create a 10x preprocessing-only config for the lung NextGen checkpoint."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preprocessing = config["preprocessing"]
    preprocessing["output_dir"] = str(NEXTGEN_10X_FEATURE_ROOT)
    preprocessing["mira_seg_checkpoint_path"] = str(NEXTGEN_CHECKPOINT)
    # The checkpoint specifies 512 px patches. At 10x (1.0 um/px), each tile
    # therefore spans 512 um and reaches the encoder without another resize.
    preprocessing["tile_size_px"] = 512
    preprocessing["tile_size_um"] = 512.0
    NEXTGEN_10X_CONFIG_PATH.write_text(
        yaml.safe_dump({"preprocessing": preprocessing}, sort_keys=False)
    )


def prepare_imagenet_config() -> None:
    """Create a preprocessing-only config for ImageNet-initialized MiT-B2."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preprocessing = config["preprocessing"]
    preprocessing["output_dir"] = str(IMAGENET_FEATURE_ROOT)
    preprocessing["extractor"] = "mira_imagenet"
    preprocessing.pop("mira_seg_checkpoint_path", None)
    IMAGENET_CONFIG_PATH.write_text(
        yaml.safe_dump({"preprocessing": preprocessing}, sort_keys=False)
    )


def prepare_vicreg_config() -> None:
    """Create a preprocessing-only config for the 20x VICReg checkpoint."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preprocessing = config["preprocessing"]
    preprocessing["output_dir"] = str(VICREG_FEATURE_ROOT)
    preprocessing["extractor"] = "vicreg"
    preprocessing["vicreg_checkpoint_path"] = str(VICREG_CHECKPOINT)
    preprocessing["vicreg_batch_size"] = 64
    preprocessing.pop("mira_seg_checkpoint_path", None)
    # The checkpoint contract specifies 224 px at 0.5 um/px (20x).
    preprocessing["tile_size_px"] = 224
    preprocessing["tile_size_um"] = 112.0
    VICREG_CONFIG_PATH.write_text(
        yaml.safe_dump({"preprocessing": preprocessing}, sort_keys=False)
    )


def prepare_mil_comparison_configs() -> None:
    """Create matched five-fold MIL configs with identical patient splits."""
    base_config = yaml.safe_load(CV5_CONFIG_PATH.read_text())
    reference_splits = RESULTS / "training_cv5" / "splits.json"
    feature_dirs = {
        "mira": FEATURE_DIR,
        "ignite": IGNITE_FEATURE_DIR,
        "nextgen": NEXTGEN_FEATURE_DIR,
        "imagenet": IMAGENET_FEATURE_DIR,
        "vicreg": VICREG_FEATURE_DIR,
        "autoencoder": AUTOENCODER_FEATURE_DIR,
    }

    for name, feature_dir in feature_dirs.items():
        output_dir = MIL_COMPARISON_ROOT / name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference_splits, output_dir / "splits.json")

        config = {
            "crossval": {
                **base_config["crossval"],
                "output_dir": str(output_dir),
                "feature_dir": str(feature_dir),
            },
            "advanced_config": base_config["advanced_config"],
        }
        MIL_COMPARISON_CONFIGS[name].write_text(yaml.safe_dump(config, sort_keys=False))


def run_mil_comparison() -> None:
    """Train matched five-fold MIL models for all feature sets."""
    prepare_mil_comparison_configs()
    stamp = str(Path(sys.executable).with_name("stamp"))
    for name, config_path in MIL_COMPARISON_CONFIGS.items():
        run_stage(
            f"mil_crossval_{name}",
            [stamp, "--config", str(config_path), "crossval"],
        )
    write_status("mil_comparison", "complete")


def summarize_mil_comparison() -> None:
    """Validate matched OOF predictions and write pooled/foldwise metrics."""
    split_payload = json.loads((RESULTS / "training_cv5" / "splits.json").read_text())
    fold_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for name in MIL_COMPARISON_CONFIGS:
        prediction_frames = []
        for fold_idx, split in enumerate(split_payload["splits"]):
            predictions = pd.read_csv(
                MIL_COMPARISON_ROOT / name / f"split-{fold_idx}" / "patient-preds.csv"
            )
            expected_patients = set(split["test_patients"])
            actual_patients = set(predictions["PATIENT"])
            if actual_patients != expected_patients:
                raise RuntimeError(
                    f"{name} fold {fold_idx} does not match the reference split"
                )
            if predictions["PATIENT"].duplicated().any():
                raise RuntimeError(
                    f"{name} fold {fold_idx} contains duplicate patients"
                )

            y_true = (predictions["DIAGNOSIS"] == "LUAD").astype(int)
            y_pred = (predictions["pred"] == "LUAD").astype(int)
            y_score = predictions["DIAGNOSIS_LUAD"]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            fold_records.append(
                {
                    "feature_set": name,
                    "fold": fold_idx,
                    "n_patients": len(predictions),
                    "auroc": roc_auc_score(y_true, y_score),
                    "accuracy": accuracy_score(y_true, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                    "macro_f1": f1_score(y_true, y_pred, average="macro"),
                    "sensitivity_luad": tp / (tp + fn),
                    "specificity_lscc": tn / (tn + fp),
                }
            )
            prediction_frames.append(predictions)

        pooled = pd.concat(prediction_frames, ignore_index=True)
        if len(pooled) != 90 or pooled["PATIENT"].nunique() != 90:
            raise RuntimeError(f"{name} does not contain 90 unique OOF patients")
        y_true = (pooled["DIAGNOSIS"] == "LUAD").astype(int)
        y_pred = (pooled["pred"] == "LUAD").astype(int)
        y_score = pooled["DIAGNOSIS_LUAD"]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        set_folds = pd.DataFrame(fold_records)
        set_folds = set_folds[set_folds["feature_set"] == name]
        summary_records.append(
            {
                "feature_set": name,
                "n_patients": len(pooled),
                "pooled_auroc": roc_auc_score(y_true, y_score),
                "fold_auroc_mean": set_folds["auroc"].mean(),
                "fold_auroc_std": set_folds["auroc"].std(ddof=1),
                "pooled_accuracy": accuracy_score(y_true, y_pred),
                "pooled_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "pooled_macro_f1": f1_score(y_true, y_pred, average="macro"),
                "sensitivity_luad": tp / (tp + fn),
                "specificity_lscc": tn / (tn + fp),
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            }
        )

    pd.DataFrame(fold_records).to_csv(
        MIL_COMPARISON_ROOT / "per_fold_metrics.csv", index=False
    )
    pd.DataFrame(summary_records).sort_values("pooled_auroc", ascending=False).to_csv(
        MIL_COMPARISON_ROOT / "summary.csv", index=False
    )


def prepare_mil_test_configs() -> None:
    """Create matched deployments of all fold models on the held-out test set."""
    base_config = yaml.safe_load(CONFIG_PATH.read_text())
    feature_dirs = {
        "mira": FEATURE_DIR,
        "ignite": IGNITE_FEATURE_DIR,
        "nextgen": NEXTGEN_FEATURE_DIR,
        "imagenet": IMAGENET_FEATURE_DIR,
        "vicreg": VICREG_FEATURE_DIR,
        "autoencoder": AUTOENCODER_FEATURE_DIR,
    }

    for name, feature_dir in feature_dirs.items():
        checkpoints = [
            MIL_COMPARISON_ROOT / name / f"split-{fold_idx}" / "model.ckpt"
            for fold_idx in range(5)
        ]
        missing = [checkpoint for checkpoint in checkpoints if not checkpoint.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {name} checkpoints: {missing}")

        deployment = {
            **base_config["deployment"],
            "output_dir": str(MIL_TEST_ROOT / name),
            "checkpoint_paths": [str(checkpoint) for checkpoint in checkpoints],
            "feature_dir": str(feature_dir),
        }
        MIL_TEST_CONFIGS[name].write_text(
            yaml.safe_dump({"deployment": deployment}, sort_keys=False)
        )


def run_mil_test_deployments() -> None:
    """Apply every fold model to the independent patient test set."""
    prepare_mil_test_configs()
    stamp = str(Path(sys.executable).with_name("stamp"))
    for name, config_path in MIL_TEST_CONFIGS.items():
        run_stage(
            f"mil_test_{name}",
            [stamp, "--config", str(config_path), "deploy"],
        )
    write_status("mil_test", "complete")


def summarize_mil_test() -> None:
    """Write foldwise and mean test metrics for all feature sets."""
    expected_patients = set(pd.read_csv(RESULTS / "test_clini.csv")["PATIENT"])
    fold_records: list[dict[str, object]] = []

    for name in MIL_TEST_CONFIGS:
        for fold_idx in range(5):
            predictions = pd.read_csv(
                MIL_TEST_ROOT / name / f"patient-preds-{fold_idx}.csv"
            )
            actual_patients = set(predictions["PATIENT"])
            if actual_patients != expected_patients:
                raise RuntimeError(
                    f"{name} fold {fold_idx} does not contain the 15 test patients"
                )
            if predictions["PATIENT"].duplicated().any():
                raise RuntimeError(f"{name} fold {fold_idx} contains duplicates")

            y_true = (predictions["DIAGNOSIS"] == "LUAD").astype(int)
            y_pred = (predictions["pred"] == "LUAD").astype(int)
            y_score = predictions["DIAGNOSIS_LUAD"]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            fold_records.append(
                {
                    "feature_set": name,
                    "fold": fold_idx,
                    "n_patients": len(predictions),
                    "auroc": roc_auc_score(y_true, y_score),
                    "accuracy": accuracy_score(y_true, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                    "macro_f1": f1_score(y_true, y_pred, average="macro"),
                    "sensitivity_luad": tp / (tp + fn),
                    "specificity_lscc": tn / (tn + fp),
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_positive": tp,
                }
            )

    fold_metrics = pd.DataFrame(fold_records)
    metric_columns = [
        "auroc",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "sensitivity_luad",
        "specificity_lscc",
    ]
    summary = fold_metrics.groupby("feature_set", sort=False)[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("auroc_mean", ascending=False)

    MIL_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(MIL_TEST_ROOT / "per_fold_metrics.csv", index=False)
    summary.to_csv(MIL_TEST_ROOT / "summary_mean_std.csv", index=False)


def run_stage(stage: str, command: list[str]) -> None:
    write_status(stage, "running", command=command)
    with PIPELINE_LOG.open("a") as log:
        log.write(
            f"\n[{datetime.now().isoformat()}] START {stage}: {' '.join(command)}\n"
        )
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"[{datetime.now().isoformat()}] COMPLETE {stage}\n")
    write_status(stage, "complete")


def main() -> None:
    try:
        prepare()
        stamp = str(Path(sys.executable).with_name("stamp"))
        base = [stamp, "--config", str(CONFIG_PATH)]
        for stage, subcommand in (
            ("feature_extraction", "preprocess"),
            ("training_and_validation", "crossval"),
            ("test_deployment", "deploy"),
            ("test_statistics", "statistics"),
        ):
            run_stage(stage, [*base, subcommand])
        write_status("pipeline", "complete")
    except Exception as exc:
        write_status("pipeline", "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
