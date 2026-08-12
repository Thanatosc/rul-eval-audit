"""Fixed-budget LSTM, 1D-CNN, Transformer, and LightGBM baselines."""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_existing_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _existing_cublas_config not in (None, _CUBLAS_WORKSPACE_CONFIG):
    raise RuntimeError(
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 for registered deterministic CUDA runs"
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs)
        return self.head(sequence[:, -1, :])


class CNN1DRegressor(nn.Module):
    def __init__(self, input_size: int, channels: int = 64, output_size: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.features(inputs.transpose(1, 2)).squeeze(-1)
        return self.head(encoded)


class SinusoidalPositionEncoding(nn.Module):
    """Deterministic sinusoidal positions for chronological RUL windows."""

    def __init__(self, d_model: int, max_length: int = 512):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisors = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * divisors)
        encoding[:, 1::2] = torch.cos(positions * divisors)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence length exceeds registered positional capacity")
        return inputs + self.encoding[:, : inputs.shape[1]].to(dtype=inputs.dtype)


class TransformerRegressor(nn.Module):
    """Small fixed-budget Transformer encoder for the unified asset grid."""

    def __init__(
        self,
        input_size: int,
        *,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        feedforward_size: int = 128,
        dropout: float = 0.10,
        output_size: int = 1,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("Transformer d_model must be divisible by nhead")
        self.input_projection = nn.Linear(input_size, d_model)
        self.position_encoding = SinusoidalPositionEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(inputs)
        encoded = self.position_encoding(encoded)
        encoded = self.encoder(encoded)
        return self.head(self.output_norm(encoded[:, -1, :]))


@dataclass(frozen=True)
class NeuralFit:
    model: nn.Module
    best_val_rmse: float
    epochs_completed: int
    device: str


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("registered CUDA device requested but CUDA is unavailable")
        index = resolved.index if resolved.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"registered CUDA device index is unavailable: {device}")
        torch.cuda.set_device(index)
    return resolved


def pinball_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    quantiles = torch.tensor([0.10, 0.50, 0.90], dtype=predictions.dtype, device=predictions.device)
    errors = targets.unsqueeze(1) - predictions
    return torch.maximum(quantiles * errors, (quantiles - 1.0) * errors).mean()


def build_neural_model(model_id: str, input_size: int, output_mode: str = "point") -> nn.Module:
    output_size = 1 if output_mode == "point" else 3
    if model_id == "lstm":
        return LSTMRegressor(input_size=input_size, output_size=output_size)
    if model_id == "cnn_1d":
        return CNN1DRegressor(input_size=input_size, output_size=output_size)
    if model_id == "transformer":
        return TransformerRegressor(input_size=input_size, output_size=output_size)
    raise ValueError(f"unsupported neural model: {model_id}")


def fit_neural(
    model_id: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    output_mode: str = "point",
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    device: str = "cpu",
    inference_batch_size: int = 4096,
) -> NeuralFit:
    """Train one neural baseline and retain the best validation checkpoint."""

    set_deterministic_seed(seed)
    resolved_device = _resolve_device(device)
    model = build_neural_model(model_id, train_x.shape[-1], output_mode).to(resolved_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        pin_memory=resolved_device.type == "cuda",
    )
    validation_y = np.asarray(val_y, dtype=np.float32)
    best_state: dict[str, Any] | None = None
    best_rmse = float("inf")
    stale_epochs = 0
    epochs_completed = 0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(resolved_device, non_blocking=False)
            batch_y = batch_y.to(resolved_device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            if output_mode == "point":
                loss = nn.functional.mse_loss(outputs.squeeze(1), batch_y)
            else:
                loss = pinball_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
        epochs_completed = epoch + 1
        validation_prediction = predict_neural(
            model,
            val_x,
            output_mode=output_mode,
            batch_size=inference_batch_size,
        )
        if output_mode == "quantile":
            validation_prediction = validation_prediction[:, 1]
        validation_rmse = float(
            np.sqrt(np.mean(np.square(validation_prediction - validation_y)))
        )
        if validation_rmse < best_rmse - 1e-9:
            best_rmse = validation_rmse
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("neural training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return NeuralFit(
        model=model,
        best_val_rmse=best_rmse,
        epochs_completed=epochs_completed,
        device=str(resolved_device),
    )


def predict_neural(
    model: nn.Module,
    features: np.ndarray,
    *,
    output_mode: str = "point",
    batch_size: int = 4096,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("prediction batch_size must be positive")
    model.eval()
    parameter = next(model.parameters())
    device = parameter.device
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            inputs = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32
            ).to(device, non_blocking=False)
            batches.append(model(inputs).detach().cpu().numpy())
    if not batches:
        width = 1 if output_mode == "point" else 3
        outputs = np.empty((0, width), dtype=np.float32)
    else:
        outputs = np.concatenate(batches, axis=0)
    if output_mode == "point":
        return outputs.reshape(-1)
    return np.sort(outputs, axis=1)


def _flatten(features: np.ndarray) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError("LightGBM expects [sample, window, sensor] input")
    return features.reshape(len(features), -1)


def fit_lightgbm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    seed: int,
    output_mode: str = "point",
    n_estimators: int = 500,
) -> lgb.LGBMRegressor | dict[float, lgb.LGBMRegressor]:
    """Fit registered flattened-window LightGBM models."""

    common = {
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
        "random_state": seed,
        "n_jobs": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    flattened = _flatten(train_x)
    if output_mode == "point":
        model = lgb.LGBMRegressor(objective="regression", **common)
        model.fit(flattened, train_y)
        return model
    models: dict[float, lgb.LGBMRegressor] = {}
    for quantile in (0.10, 0.50, 0.90):
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **common)
        model.fit(flattened, train_y)
        models[quantile] = model
    return models


def predict_lightgbm(
    model: lgb.LGBMRegressor | dict[float, lgb.LGBMRegressor],
    features: np.ndarray,
    *,
    output_mode: str = "point",
) -> np.ndarray:
    flattened = _flatten(features)
    if output_mode == "point":
        if isinstance(model, dict):
            raise TypeError("point prediction received quantile models")
        return np.asarray(model.predict(flattened), dtype=float)
    if not isinstance(model, dict):
        raise TypeError("quantile prediction requires a model per quantile")
    predictions = np.column_stack([model[q].predict(flattened) for q in (0.10, 0.50, 0.90)])
    return np.sort(predictions, axis=1)
