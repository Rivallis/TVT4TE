"""Model pool registry for the TVT benchmark.

Registers all 26 pretrained models (PTMs) used in the benchmark, covering:

* 11 supervised CNNs (ImageNet-1k, standard ``timm`` checkpoints)
* 10 self-supervised ResNet-50 checkpoints (custom weight URLs)
* 5 ViT-Base models (supervised, and SSL: MoCo v3, DINO, MAE, CLIP)

Feature-extraction layer convention (§9.6)
------------------------------------------
**Open item:** the manuscript does not specify exactly which layer is used for
each backbone family.  The defaults below are the natural penultimate
representations:

* **ResNet / DenseNet / other CNNs** — global average-pooled output of the
  final convolutional block (the ``timm`` ``forward_features`` API).
* **ViT-Base** — CLS token from the final transformer block.
* **CLIP ViT-Base** — the ``encode_image`` output (normalised in CLIP's
  native API; not renormalized here).

To override the layer for a specific model, pass ``layer=<name>`` to
:func:`tvt.extract.extract_features`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = ["MODELS", "ModelInfo", "get_model"]


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for one pretrained model.

    Parameters
    ----------
    name:
        Canonical model key.
    family:
        Backbone family — one of ``"cnn"`` (supervised), ``"ssl_resnet50"``,
        ``"vit"``.
    timm_name:
        ``timm`` model identifier, or ``None`` if the checkpoint must be loaded
        manually.
    weights_url:
        Direct URL to model weights (where applicable).
    pretraining:
        Pretraining scheme (``"supervised"``, ``"BYOL"``, etc.).
    pretraining_data:
        Dataset used for pretraining.
    feature_layer:
        Which layer to extract features from (``"global_pool"``, ``"cls_token"``).
    notes:
        Free-text notes.
    """

    name: str
    family: str
    timm_name: Optional[str]
    weights_url: Optional[str]
    pretraining: str
    pretraining_data: str
    feature_layer: str
    notes: str = ""


MODELS: Dict[str, ModelInfo] = {
    # ------------------------------------------------------------------ #
    # Supervised CNNs                                                      #
    # ------------------------------------------------------------------ #
    "resnet34": ModelInfo(
        name="resnet34",
        family="cnn",
        timm_name="resnet34",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "resnet50": ModelInfo(
        name="resnet50",
        family="cnn",
        timm_name="resnet50",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "resnet101": ModelInfo(
        name="resnet101",
        family="cnn",
        timm_name="resnet101",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "resnet152": ModelInfo(
        name="resnet152",
        family="cnn",
        timm_name="resnet152",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "densenet121": ModelInfo(
        name="densenet121",
        family="cnn",
        timm_name="densenet121",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "densenet169": ModelInfo(
        name="densenet169",
        family="cnn",
        timm_name="densenet169",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "densenet201": ModelInfo(
        name="densenet201",
        family="cnn",
        timm_name="densenet201",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "mnasnet_a1": ModelInfo(
        name="mnasnet_a1",
        family="cnn",
        timm_name="mnasnet_a1",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
        notes="MNet-A1 (MNASNet-A1).",
    ),
    "mobilenetv2": ModelInfo(
        name="mobilenetv2",
        family="cnn",
        timm_name="mobilenetv2_100",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "googlenet": ModelInfo(
        name="googlenet",
        family="cnn",
        timm_name="googlenet",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "inceptionv3": ModelInfo(
        name="inceptionv3",
        family="cnn",
        timm_name="inception_v3",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    # ------------------------------------------------------------------ #
    # Self-supervised ResNet-50                                            #
    # ------------------------------------------------------------------ #
    "byol": ModelInfo(
        name="byol",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="BYOL",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
        notes="Weights must be obtained from the official BYOL release.",
    ),
    "deepclusterv2": ModelInfo(
        name="deepclusterv2",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="DeepCluster-v2",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "infomin": ModelInfo(
        name="infomin",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="InfoMin",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "mocov1": ModelInfo(
        name="mocov1",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="MoCo-v1",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "mocov2": ModelInfo(
        name="mocov2",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="MoCo-v2",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "insdis": ModelInfo(
        name="insdis",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="Instance Discrimination",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "pclv1": ModelInfo(
        name="pclv1",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="PCL-v1",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "pclv2": ModelInfo(
        name="pclv2",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="PCL-v2",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "selav2": ModelInfo(
        name="selav2",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="SeLa-v2",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    "swav": ModelInfo(
        name="swav",
        family="ssl_resnet50",
        timm_name=None,
        weights_url=None,
        pretraining="SwAV",
        pretraining_data="ImageNet-1k",
        feature_layer="global_pool",
    ),
    # ------------------------------------------------------------------ #
    # ViT-Base (all ViT-Base/16 unless noted)                             #
    # ------------------------------------------------------------------ #
    "vit_timm": ModelInfo(
        name="vit_timm",
        family="vit",
        timm_name="vit_base_patch16_224",
        weights_url=None,
        pretraining="supervised",
        pretraining_data="ImageNet-1k",
        feature_layer="cls_token",
        notes="Standard supervised ViT-Base from timm.",
    ),
    "mocov3_vit": ModelInfo(
        name="mocov3_vit",
        family="vit",
        timm_name=None,
        weights_url="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
        pretraining="MoCo-v3",
        pretraining_data="ImageNet-1k",
        feature_layer="cls_token",
    ),
    "dino_vit": ModelInfo(
        name="dino_vit",
        family="vit",
        timm_name=None,
        weights_url="https://github.com/facebookresearch/dino",
        pretraining="DINO",
        pretraining_data="ImageNet-1k",
        feature_layer="cls_token",
    ),
    "mae_vit": ModelInfo(
        name="mae_vit",
        family="vit",
        timm_name=None,
        weights_url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        pretraining="MAE",
        pretraining_data="ImageNet-1k",
        feature_layer="cls_token",
        notes="MAE fine-tuning uses 100 epochs (vs 50 for other ViTs).",
    ),
    "clip_vit": ModelInfo(
        name="clip_vit",
        family="vit",
        timm_name=None,
        weights_url="https://github.com/mlfoundations/open_clip",
        pretraining="CLIP",
        pretraining_data="LAION-2B",
        feature_layer="cls_token",
        notes="Loaded via open_clip; encode_image output used as features.",
    ),
}


def get_model(name: str) -> ModelInfo:
    """Return the :class:`ModelInfo` record for *name*.

    Parameters
    ----------
    name:
        Model key (case-insensitive).

    Returns
    -------
    ModelInfo

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    key = name.lower()
    if key not in MODELS:
        raise KeyError(
            f"Unknown model {name!r}.  "
            f"Available: {sorted(MODELS.keys())}"
        )
    return MODELS[key]
