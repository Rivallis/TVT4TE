"""Feature extraction utilities for TVT.

Extracts frozen penultimate features from pretrained models.

Feature layer convention (§9.6)
--------------------------------
* **CNN backbones (ResNet, DenseNet, etc.)** — global average-pooled output of
  the last convolutional stage via ``timm``'s ``forward_features`` API.
* **ViT-Base (timm / MoCo v3 / DINO / MAE)** — CLS token from the final
  transformer block (index 0 of the output sequence).
* **CLIP ViT-Base** — ``encode_image`` output from ``open_clip``.

Override the layer by passing ``layer="global_pool"`` or ``layer="cls_token"``
explicitly.

**Note:** this module requires the optional heavy dependencies (``timm``,
``torch``, ``torchvision``).  They are listed under ``[extras]`` in
``pyproject.toml`` and are not installed by default.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["extract_features"]


def extract_features(
    model_name: str,
    dataset_name: str,
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 4,
    layer: Optional[str] = None,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frozen features for *model_name* on *dataset_name*.

    Parameters
    ----------
    model_name:
        Registered model key (see :mod:`tvt.benchmark.models`).
    dataset_name:
        Registered dataset key (see :mod:`tvt.benchmark.datasets`).
    data_root:
        Path to the root directory that contains dataset subdirectories.
    batch_size:
        Mini-batch size for the forward pass.
    num_workers:
        DataLoader worker count.
    layer:
        Override the default feature layer.  One of ``"global_pool"`` or
        ``"cls_token"``.
    device:
        PyTorch device string.

    Returns
    -------
    features : np.ndarray
        Float32 feature matrix ``(N, d)``.
    labels : np.ndarray
        Integer label vector ``(N,)``.
    """
    try:
        import torch
        import timm
    except ImportError as exc:
        raise ImportError(
            "Feature extraction requires 'timm' and 'torch'.  "
            "Install them with: pip install tvt4te[extract]"
        ) from exc

    from .benchmark.models import get_model

    model_info = get_model(model_name)
    _layer = layer or model_info.feature_layer

    logger.info("Loading model %s (timm: %s)", model_name, model_info.timm_name)

    if model_info.timm_name is not None:
        model = timm.create_model(
            model_info.timm_name, pretrained=True, num_classes=0
        )
    else:
        raise NotImplementedError(
            f"Custom weight loading for {model_name!r} is not yet implemented.  "
            "Download the weights and load them manually, then call "
            "extract_features_from_model()."
        )

    model = model.to(device).eval()

    # Build dataset + loader (minimal implementation; extend as needed).
    loader = _build_loader(dataset_name, data_root, batch_size, num_workers)

    all_features, all_labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            if _layer == "cls_token":
                feats = model.forward_features(images)[:, 0]
            else:
                feats = model(images)
            all_features.append(feats.cpu().float().numpy())
            all_labels.append(targets.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def _build_loader(dataset_name: str, data_root: str, batch_size: int, num_workers: int):
    """Minimal DataLoader builder.  Extend to support all 11 datasets."""
    try:
        import torch
        from torchvision import transforms, datasets
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for the default DataLoader builder."
        ) from exc

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    import os

    dataset_path = os.path.join(data_root, dataset_name)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
