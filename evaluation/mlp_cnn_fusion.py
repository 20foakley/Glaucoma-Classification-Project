from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 12


class EmbeddingFusionHead(nn.Module):
    def __init__(self, tabular_size: int, image_size: int = 512) -> None:
        super().__init__()
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_size + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, image_embedding: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        tabular_embedding = self.tabular_encoder(tabular)
        fused = torch.cat((image_embedding, tabular_embedding), dim=1)
        return self.classifier(fused).squeeze(1)


def _normalize_images(images: np.ndarray, indices: np.ndarray, stats: dict) -> np.ndarray:
    raw = np.asarray(images[indices], dtype=np.float32)
    valid = raw > 0
    clean = np.clip(np.where(valid, raw, 0), 0, float(stats["clip"]))
    normalized = np.where(valid, (clean - float(stats["mu"])) / float(stats["sd"]), 0)
    return normalized[:, None, :, :].astype(np.float32)


def _extract_image_embeddings(
    encoder: nn.Module,
    images: np.ndarray,
    indices: np.ndarray,
    stats: dict,
    device: torch.device,
) -> np.ndarray:
    embeddings = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            batch = _normalize_images(images, batch_indices, stats)
            tensor = torch.from_numpy(batch).to(device)
            embeddings.append(encoder(tensor).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        in_bin = (
            (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
            if index == 0
            else (probabilities > edges[index]) & (probabilities <= edges[index + 1])
        )
        if in_bin.any():
            accuracy = np.mean((probabilities[in_bin] >= 0.5) == labels[in_bin])
            confidence = np.mean(np.maximum(probabilities[in_bin], 1 - probabilities[in_bin]))
            result += in_bin.mean() * abs(accuracy - confidence)
    return float(result)


def _auc_ci(
    labels: np.ndarray, probabilities: np.ndarray, n_boot: int = 2000
) -> tuple[float, float]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUROC and its confidence interval require both label classes.")
    rng = np.random.default_rng(SEED)
    aucs = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        sample = np.concatenate(
            (
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            )
        )
        aucs[iteration] = roc_auc_score(labels[sample], probabilities[sample])
    lower, upper = np.percentile(aucs, [2.5, 97.5])
    return float(lower), float(upper)


def run_fusion(
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    output_dir = output_dir or project_root
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(processed_dir / "meta.csv")
    if "filename" not in metadata.columns and "file" in metadata.columns:
        metadata = metadata.rename(columns={"file": "filename"})
    features = pd.read_csv(processed_dir / "rnflt_features.csv")
    if "filename" not in metadata.columns or "filename" not in features.columns:
        raise ValueError("Metadata and RNFLT feature CSVs must contain a filename key.")

    data = metadata.merge(features, on="filename", how="inner", validate="one_to_one", sort=False)
    if len(data) != len(metadata) or len(data) != len(features):
        raise ValueError("Metadata and RNFLT feature filenames/counts do not match.")
    if not np.array_equal(data["filename"].to_numpy(), metadata["filename"].to_numpy()):
        raise ValueError("Feature alignment changed the metadata/image row order.")

    data["split"] = data["split"].astype(str).str.strip().str.lower()
    data["label"] = pd.to_numeric(data["glaucoma"], errors="coerce")
    data["race"] = (
        data["race"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "white or caucasian": "white",
                "black or african american": "black",
            }
        )
    )
    if data["label"].isna().any() or not data["label"].isin([0, 1]).all():
        raise ValueError("glaucoma must contain only binary 0/1 labels.")
    if set(data["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Expected official train, val, and test splits.")

    tabular_columns = [
        "age",
        "hcdr",
        "rim_px",
        "rnfl_mean",
        "rnfl_p10",
        "ring_mean",
        "ring_sup",
        "ring_inf",
        "ring_horiz",
        "si_asym",
    ]
    data[tabular_columns] = data[tabular_columns].apply(pd.to_numeric, errors="coerce")
    images = np.load(processed_dir / "rnflt.npy", mmap_mode="r")
    if len(images) != len(data):
        raise ValueError("RNFLT map and metadata row counts do not match.")
    if images.ndim != 3 or images.shape[1:] != (200, 200):
        raise ValueError(f"Expected RNFLT maps with shape (N, 200, 200), got {images.shape}.")
    with (processed_dir / "norm_stats.json").open() as stats_file:
        stats = json.load(stats_file)

    split_indices = {
        split: np.flatnonzero(data["split"].to_numpy() == split)
        for split in ("train", "val", "test")
    }
    train_idx, val_idx, test_idx = (
        split_indices["train"],
        split_indices["val"],
        split_indices["test"],
    )
    if len(train_idx) != 2100 or len(val_idx) != 300 or len(test_idx) != 900:
        raise ValueError("Expected official split sizes train=2100, val=300, test=900.")

    checkpoint_path = project_root / "results" / "runs_full" / "resnet18_seed0.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained CNN checkpoint not found: {checkpoint_path}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    cnn = torchvision.models.resnet18(weights=None)
    cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    cnn.fc = nn.Linear(cnn.fc.in_features, 1)
    cnn.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    cnn.fc = nn.Identity()
    cnn = cnn.to(device).eval()
    for parameter in cnn.parameters():
        parameter.requires_grad = False

    print(f"Extracting frozen CNN embeddings on {device}...")
    train_image_embeddings = _extract_image_embeddings(cnn, images, train_idx, stats, device)
    val_image_embeddings = _extract_image_embeddings(cnn, images, val_idx, stats, device)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_tabular = scaler.fit_transform(
        imputer.fit_transform(data.loc[train_idx, tabular_columns])
    ).astype(np.float32)
    val_tabular = scaler.transform(
        imputer.transform(data.loc[val_idx, tabular_columns])
    ).astype(np.float32)
    train_labels = data.loc[train_idx, "label"].to_numpy(dtype=np.float32)
    val_labels = data.loc[val_idx, "label"].to_numpy(dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_image_embeddings),
            torch.from_numpy(train_tabular),
            torch.from_numpy(train_labels),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = EmbeddingFusionHead(tabular_size=len(tabular_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    val_image_tensor = torch.from_numpy(val_image_embeddings).to(device)
    val_tabular_tensor = torch.from_numpy(val_tabular).to(device)
    val_label_tensor = torch.from_numpy(val_labels).to(device)

    best_loss, best_state, best_epoch, stale_epochs = float("inf"), None, 0, 0
    start_time = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for image_batch, tabular_batch, label_batch in train_loader:
            image_batch = image_batch.to(device)
            tabular_batch = tabular_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image_batch, tabular_batch), label_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_loss = criterion(
                model(val_image_tensor, val_tabular_tensor), val_label_tensor
            ).item()
        print(f"Epoch {epoch:03d} | validation BCE: {val_loss:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid fusion checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_probabilities = (
            torch.sigmoid(model(val_image_tensor, val_tabular_tensor)).cpu().numpy()
        )

    torch.save(
        {
            "fusion_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "cnn_checkpoint": str(checkpoint_path.relative_to(project_root)),
            "tabular_columns": tabular_columns,
            "tabular_imputer_median": imputer.statistics_.tolist(),
            "tabular_scaler_mean": scaler.mean_.tolist(),
            "tabular_scaler_scale": scaler.scale_.tolist(),
            "image_normalization": stats,
            "best_epoch": best_epoch,
            "seed": SEED,
        },
        output_dir / "mlp_cnn_fusion.pt",
    )
    print(
        f"Selected epoch {best_epoch} by validation BCE in "
        f"{time.perf_counter() - start_time:.1f}s"
    )

    # Keep test evaluation after model/checkpoint selection.
    test_image_embeddings = _extract_image_embeddings(cnn, images, test_idx, stats, device)
    test_tabular = scaler.transform(
        imputer.transform(data.loc[test_idx, tabular_columns])
    ).astype(np.float32)
    test_labels = data.loc[test_idx, "label"].to_numpy(dtype=np.int64)
    with torch.inference_mode():
        test_probabilities = (
            torch.sigmoid(
                model(
                    torch.from_numpy(test_image_embeddings).to(device),
                    torch.from_numpy(test_tabular).to(device),
                )
            )
            .cpu()
            .numpy()
        )

    prediction_rows = []
    metric_rows = []
    for split_name, indices, labels, probabilities in (
        ("Validation", val_idx, val_labels.astype(np.int64), val_probabilities),
        ("Test", test_idx, test_labels, test_probabilities),
    ):
        races = data.loc[indices, "race"].to_numpy()
        filenames = data.loc[indices, "filename"].to_numpy()
        prediction_rows.extend(
            {
                "filename": filename,
                "split": split_name,
                "race": race,
                "label": int(label),
                "probability": float(probability),
                "predicted_label": int(probability >= 0.5),
            }
            for filename, race, label, probability in zip(
                filenames, races, labels, probabilities
            )
        )
        for race in ("white", "black", "asian", "overall"):
            subset = np.ones(len(labels), dtype=bool) if race == "overall" else races == race
            subset_labels, subset_probabilities = labels[subset], probabilities[subset]
            if np.unique(subset_labels).size != 2:
                raise ValueError(f"{split_name}/{race} subset must contain both label classes.")
            lower, upper = _auc_ci(subset_labels, subset_probabilities)
            metric_rows.append(
                {
                    "model": "Frozen ResNet-18 + tabular MLP embedding fusion",
                    "split": split_name,
                    "race": race.title(),
                    "AUROC": roc_auc_score(subset_labels, subset_probabilities),
                    "AUROC_CI_95_Lower": lower,
                    "AUROC_CI_95_Upper": upper,
                    "ECE_10_bins": _ece(subset_labels, subset_probabilities),
                    "n": int(subset.sum()),
                    "glaucoma_cases": int(subset_labels.sum()),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(output_dir / "mlp_cnn_fusion_predictions.csv", index=False)
    metrics.to_csv(output_dir / "mlp_cnn_fusion_results.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"Saved predictions and metrics under {output_dir.resolve()}")
    return metrics


if __name__ == "__main__":
    run_fusion()
