"""Governed Python boundary for PhysioML foundation embeddings."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def versions() -> dict[str, str]:
    import torch

    return {
        "numpy": _version("numpy"),
        "python_torch": str(torch.__version__),
        "safetensors": _version("safetensors"),
    }


def download_snapshot(model_id: str, revision: str, destination: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=destination,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
        ],
    )
    shutil.rmtree(Path(destination) / ".cache", ignore_errors=True)


def _tiny_embed(
    asset_dir: Path,
    data: np.ndarray,
    reduction: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from safetensors.torch import load_file

    with (asset_dir / "config.json").open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config != {
        "adapter": "physioml_tiny",
        "feature_order": ["mean", "sd", "minimum", "maximum"],
        "input_axes": "batch,channel,time",
        "minimum_time_samples": 8,
        "output_width": 6,
        "schema_version": "1.0.0",
    }:
        raise RuntimeError("tiny foundation config does not match schema 1.0.0")
    tensors = load_file(str(asset_dir / "model.safetensors"), device="cpu")
    if sorted(tensors) != ["bias", "projection"]:
        raise RuntimeError("tiny safetensors checkpoint has unexpected tensors")
    projection = tensors["projection"]
    bias = tensors["bias"]
    if tuple(projection.shape) != (4, 6) or tuple(bias.shape) != (6,):
        raise RuntimeError("tiny safetensors checkpoint has invalid dimensions")

    array = np.asarray(data, dtype=np.float32, order="C")
    if array.ndim != 3:
        raise ValueError("foundation input must use batch x channel x time axes")
    if array.shape[2] < int(config["minimum_time_samples"]):
        raise ValueError(
            "foundation input last axis is not the documented time axis"
        )
    if not np.isfinite(array).all():
        raise ValueError("foundation input contains non-finite values")
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for first in range(0, array.shape[0], batch_size):
            value = torch.from_numpy(
                np.ascontiguousarray(array[first : first + batch_size])
            )
            mean = value.mean(dim=2)
            sd = value.std(dim=2, unbiased=False)
            minimum = value.amin(dim=2)
            maximum = value.amax(dim=2)
            features = torch.stack([mean, sd, minimum, maximum], dim=2)
            embedded = features @ projection + bias
            batches.append(embedded)
        output = torch.cat(batches, dim=0)
        if reduction == "mean":
            output = output.mean(dim=1)
        elif reduction != "none":
            raise ValueError("unknown embedding reduction")
    result = output.detach().cpu().numpy().astype(np.float64, copy=False)
    if result.shape[0] != array.shape[0] or not np.isfinite(result).all():
        raise RuntimeError("tiny adapter violated the output contract")
    return result


def _moment_embed(
    asset_dir: Path,
    data: np.ndarray,
    reduction: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        str(asset_dir),
        model_kwargs={"task_name": "embedding"},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.init()
    model.eval()
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for first in range(0, data.shape[0], batch_size):
            value = torch.from_numpy(
                np.ascontiguousarray(
                    data[first : first + batch_size],
                    dtype=np.float32,
                )
            )
            output = model(x_enc=value)
            if not hasattr(output, "embeddings"):
                raise RuntimeError("MOMENT output has no documented embeddings field")
            batches.append(output.embeddings.detach().cpu())
    embedded = torch.cat(batches, dim=0)
    if reduction == "mean" and embedded.ndim == 3:
        embedded = embedded.mean(dim=1)
    elif reduction == "none" and embedded.ndim != 3:
        raise RuntimeError("MOMENT unreduced output has no documented token axis")
    if reduction == "mean" and embedded.ndim != 2:
        raise RuntimeError("MOMENT mean embedding is not rank two")
    return embedded.numpy().astype(np.float64, copy=False)


def embed(
    asset_dir: str,
    adapter: str,
    model_id: str,
    data: Any,
    reduction: str,
    batch_size: int,
) -> dict[str, Any]:
    path = Path(asset_dir).resolve(strict=True)
    array = np.asarray(data, dtype=np.float32, order="C")
    if model_id == "physioml/foundation-tiny":
        output = _tiny_embed(path, array, reduction, int(batch_size))
    elif adapter == "moment":
        output = _moment_embed(path, array, reduction, int(batch_size))
    else:
        raise RuntimeError(
            f"{adapter} has no CPU-verified safe embedding adapter"
        )
    return {
        "embeddings": output,
        "versions": versions(),
    }
