"""Definitions extracted from authoritative FINAL V2 cells; see source_definitions.json."""
from pathlib import Path
import copy, gc, json, random, re, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from PIL import Image, ImageFilter
# Publication module defaults; original full runner selects its backend explicitly.
device = torch.device("cpu")
HIDDEN_DIM = 256
DROPOUT_RATE = 0.30


def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_tensor_loader(
    feature_arrays,
    labels,
    batch_size,
    shuffle,
    seed,
):
    feature_tensors = [
        torch.from_numpy(
            np.asarray(
                feature_array,
                dtype=np.float32
            )
        )
        for feature_array in feature_arrays
    ]

    label_tensor = torch.from_numpy(
        np.asarray(
            labels,
            dtype=np.float32
        )
    )

    dataset = TensorDataset(
        *feature_tensors,
        label_tensor
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=(
            generator
            if shuffle
            else None
        ),
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )


def move_batch_to_device(batch):
    *feature_tensors, label_tensor = batch

    feature_tensors = [
        feature_tensor.to(
            device=device,
            dtype=torch.float32
        )
        for feature_tensor in feature_tensors
    ]

    label_tensor = label_tensor.to(
        device=device,
        dtype=torch.float32
    )

    return feature_tensors, label_tensor


def count_trainable_parameters(model):
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def train_binary_model(
    model,
    train_loader,
    validation_loader,
    learning_rate,
    weight_decay,
    maximum_epochs,
    patience,
):
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_function = (
        torch.nn.BCEWithLogitsLoss()
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = None

    history_rows = []

    for epoch in range(
        1,
        maximum_epochs + 1
    ):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for batch in train_loader:
            features, labels = (
                move_batch_to_device(batch)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                *features
            ).reshape(-1)

            loss = loss_function(
                logits,
                labels
            )

            loss.backward()
            optimizer.step()

            batch_size = int(
                labels.shape[0]
            )

            total_train_loss += (
                float(loss.item())
                * batch_size
            )

            total_train_samples += (
                batch_size
            )

        mean_train_loss = (
            total_train_loss
            / total_train_samples
        )

        model.eval()

        total_validation_loss = 0.0
        total_validation_samples = 0

        with torch.inference_mode():
            for batch in validation_loader:
                features, labels = (
                    move_batch_to_device(
                        batch
                    )
                )

                logits = model(
                    *features
                ).reshape(-1)

                loss = loss_function(
                    logits,
                    labels
                )

                batch_size = int(
                    labels.shape[0]
                )

                total_validation_loss += (
                    float(loss.item())
                    * batch_size
                )

                total_validation_samples += (
                    batch_size
                )

        mean_validation_loss = (
            total_validation_loss
            / total_validation_samples
        )

        improved = (
            mean_validation_loss
            < best_validation_loss
            - 1e-6
        )

        if improved:
            best_validation_loss = (
                mean_validation_loss
            )

            best_epoch = epoch

            best_state_dict = copy.deepcopy(
                {
                    key: value
                    .detach()
                    .cpu()
                    .clone()
                    for key, value
                    in model.state_dict().items()
                }
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        history_rows.append({
            "epoch": epoch,
            "train_loss": (
                mean_train_loss
            ),
            "validation_loss": (
                mean_validation_loss
            ),
            "is_best_epoch": improved,
        })

        if (
            epochs_without_improvement
            >= patience
        ):
            break

    if best_state_dict is None:
        raise RuntimeError(
            "Training tidak menghasilkan "
            "best model state."
        )

    model.load_state_dict(
        best_state_dict
    )

    model = model.to(device)
    model.eval()

    training_history_df = pd.DataFrame(
        history_rows
    )

    training_summary = {
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "epochs_completed": int(
            len(training_history_df)
        ),
        "early_stopped": bool(
            len(training_history_df)
            < maximum_epochs
        ),
    }

    return (
        model,
        training_history_df,
        training_summary,
    )


def predict_binary_logits(
    model,
    data_loader,
):
    model.eval()

    collected_logits = []
    collected_labels = []

    with torch.inference_mode():
        for batch in data_loader:
            features, labels = (
                move_batch_to_device(batch)
            )

            logits = model(
                *features
            ).reshape(-1)

            collected_logits.append(
                logits.cpu().numpy()
            )

            collected_labels.append(
                labels.cpu().numpy()
            )

    logits_array = np.concatenate(
        collected_logits
    ).astype(np.float32)

    labels_array = np.concatenate(
        collected_labels
    ).astype(np.float32)

    return logits_array, labels_array


def sigmoid_numpy(logits):
    logits = np.asarray(
        logits,
        dtype=np.float64
    )

    return 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                logits,
                -50,
                50
            )
        )
    )


def select_validation_threshold(
    labels,
    probabilities,
    threshold_candidates,
):
    threshold_rows = []

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    for threshold in threshold_candidates:
        predictions = (
            probabilities >= threshold
        ).astype(np.int64)

        threshold_rows.append({
            "threshold": float(
                threshold
            ),
            "f1": float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),
            "precision": float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),
            "accuracy": float(
                accuracy_score(
                    labels,
                    predictions
                )
            ),
        })

    threshold_search_df = pd.DataFrame(
        threshold_rows
    )

    best_row = (
        threshold_search_df
        .sort_values(
            [
                "f1",
                "precision",
                "recall",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ]
        )
        .iloc[0]
    )

    best_threshold = float(
        best_row["threshold"]
    )

    return (
        best_threshold,
        threshold_search_df,
    )


def calculate_binary_metrics(
    labels,
    probabilities,
    threshold,
):
    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    confusion = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    true_negative = int(
        confusion[0, 0]
    )

    false_positive = int(
        confusion[0, 1]
    )

    false_negative = int(
        confusion[1, 0]
    )

    true_positive = int(
        confusion[1, 1]
    )

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities
            )
        ),
        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities
            )
        ),
        "brier_score": float(
            brier_score_loss(
                labels,
                probabilities
            )
        ),
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
    }


class TextOnlyMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                input_dim,
                hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                dropout
            ),
            torch.nn.Linear(
                hidden_dim,
                1
            ),
        )

    def forward(self, text_embeddings):
        return self.classifier(
            text_embeddings
        ).squeeze(-1)


def load_torch_checkpoint(
    checkpoint_path
):
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location="cpu"
        )


class ImageOnlyMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=1152,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                input_dim,
                hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                dropout
            ),
            torch.nn.Linear(
                hidden_dim,
                1
            ),
        )

    def forward(self, image_embeddings):
        return self.classifier(
            image_embeddings
        ).squeeze(-1)


class LearnableWeightedLogitFusion(
    torch.nn.Module
):
    def __init__(self):
        super().__init__()

        # softmax([0, 0]) = [0.5, 0.5]
        self.raw_weights = torch.nn.Parameter(
            torch.zeros(2)
        )

    def normalized_weights(self):
        return torch.softmax(
            self.raw_weights,
            dim=0
        )

    def forward(
        self,
        text_logits,
        image_logits,
    ):
        weights = self.normalized_weights()

        return (
            weights[0]
            * text_logits.reshape(-1)
            +
            weights[1]
            * image_logits.reshape(-1)
        )


def load_component_model(
    model_type,
    checkpoint_path,
):
    checkpoint = load_torch_checkpoint(
        checkpoint_path
    )

    if model_type == "text":
        model = TextOnlyMLP(
            input_dim=1024,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT_RATE,
        )

    elif model_type == "image":
        model = ImageOnlyMLP(
            input_dim=1152,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT_RATE,
        )

    else:
        raise ValueError(
            f"Model type tidak dikenal: "
            f"{model_type}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def get_component_logits(
    model,
    embeddings,
    labels,
    seed,
):
    loader = create_tensor_loader(
        feature_arrays=[embeddings],
        labels=labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )

    logits, returned_labels = (
        predict_binary_logits(
            model,
            loader
        )
    )

    if not np.array_equal(
        returned_labels.astype(np.int64),
        np.asarray(
            labels,
            dtype=np.int64
        )
    ):
        raise ValueError(
            "Urutan label berubah."
        )

    return logits.astype(np.float32)


class RawConcatenationFusion(torch.nn.Module):
    def __init__(
        self,
        text_dim=1024,
        image_dim=1152,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                text_dim + image_dim,
                hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                hidden_dim,
                1
            ),
        )

    def forward(
        self,
        text_embeddings,
        image_embeddings,
    ):
        combined_embeddings = torch.cat(
            [
                text_embeddings,
                image_embeddings,
            ],
            dim=1
        )

        return self.classifier(
            combined_embeddings
        ).squeeze(-1)


class ProjectionFusion(torch.nn.Module):
    def __init__(
        self,
        text_dim=1024,
        image_dim=1152,
        projection_dim=512,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(
                text_dim,
                projection_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

        self.image_projection = torch.nn.Sequential(
            torch.nn.Linear(
                image_dim,
                projection_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                projection_dim * 2,
                hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                hidden_dim,
                1
            ),
        )

    def forward(
        self,
        text_embeddings,
        image_embeddings,
    ):
        text_features = self.text_projection(
            text_embeddings
        )

        image_features = self.image_projection(
            image_embeddings
        )

        fused_features = torch.cat(
            [
                text_features,
                image_features,
            ],
            dim=1
        )

        return self.classifier(
            fused_features
        ).squeeze(-1)


class ContrastiveSimilarityFusion(
    torch.nn.Module
):
    def __init__(
        self,
        text_dim=1024,
        image_dim=1152,
        projection_dim=512,
    ):
        super().__init__()

        self.text_projection = torch.nn.Linear(
            text_dim,
            projection_dim
        )

        self.image_projection = torch.nn.Linear(
            image_dim,
            projection_dim
        )

        # exp(log(1 / 0.07)) = 14.2857
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(
                np.log(1 / 0.07),
                dtype=torch.float32
            )
        )

    def forward(
        self,
        text_embeddings,
        image_embeddings,
    ):
        text_features = F.normalize(
            self.text_projection(
                text_embeddings
            ),
            p=2,
            dim=1
        )

        image_features = F.normalize(
            self.image_projection(
                image_embeddings
            ),
            p=2,
            dim=1
        )

        cosine_similarity = torch.sum(
            text_features * image_features,
            dim=1
        )

        # Pembatasan mengikuti praktik umum
        # agar scale tidak membesar tanpa batas.
        scale = (
            self.logit_scale
            .exp()
            .clamp(max=100.0)
        )

        return scale * cosine_similarity
