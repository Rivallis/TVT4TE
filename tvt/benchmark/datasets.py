"""Dataset registry for the TVT benchmark.

Registers all 11 benchmark datasets with their official download URLs, split
sizes, number of classes, and task regime.

**Data redistribution:** none of these datasets may be redistributed.  Use the
download helper (:func:`download_dataset`) to fetch each dataset into a
user-specified root directory.

**SUN397 note:** SUN397 does not have an official benchmark test split; the
``test_size`` entry is ``None`` and the benchmark evaluation protocol uses only
the training portion.  Attempting to evaluate on the SUN397 test split will
raise a ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["DATASETS", "DatasetInfo", "get_dataset"]

_REGIME_ID = "ID"
_REGIME_OOD = "OOD"
_REGIME_FG = "FG"


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata record for one benchmark dataset.

    Parameters
    ----------
    name:
        Canonical dataset name used as a key.
    train_size:
        Number of training images.
    test_size:
        Number of test images, or ``None`` if no official test split is used.
    num_classes:
        Number of output categories.
    regime:
        Task regime relative to ImageNet — one of ``"ID"``, ``"OOD"``,
        ``"FG"``.
    url:
        Authoritative download URL.
    notes:
        Optional free-text notes.
    """

    name: str
    train_size: int
    test_size: Optional[int]
    num_classes: int
    regime: str
    url: str
    notes: str = ""


DATASETS: Dict[str, DatasetInfo] = {
    "aircraft": DatasetInfo(
        name="aircraft",
        train_size=3334,
        test_size=3333,
        num_classes=100,
        regime=_REGIME_FG,
        url="https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/",
        notes="FGVC-Aircraft; variant-level labels used in the benchmark.",
    ),
    "cars": DatasetInfo(
        name="cars",
        train_size=8144,
        test_size=8041,
        num_classes=196,
        regime=_REGIME_FG,
        url="https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
        notes="Stanford Cars.",
    ),
    "food": DatasetInfo(
        name="food",
        train_size=80800,
        test_size=20200,
        num_classes=101,
        regime=_REGIME_FG,
        url="https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
        notes="Food-101.",
    ),
    "pets": DatasetInfo(
        name="pets",
        train_size=3680,
        test_size=3669,
        num_classes=37,
        regime=_REGIME_FG,
        url="https://www.robots.ox.ac.uk/~vgg/data/pets/",
        notes="Oxford-IIIT Pet.",
    ),
    "flowers": DatasetInfo(
        name="flowers",
        train_size=1020,
        test_size=6169,
        num_classes=102,
        regime=_REGIME_FG,
        url="https://www.robots.ox.ac.uk/~vgg/data/flowers/",
        notes="Oxford-102 Flowers.",
    ),
    "caltech101": DatasetInfo(
        name="caltech101",
        train_size=7315,
        test_size=1829,
        num_classes=101,
        regime=_REGIME_ID,
        url="https://data.caltech.edu/records/mzrjq-6wc02",
        notes="Caltech-101.",
    ),
    "cifar10": DatasetInfo(
        name="cifar10",
        train_size=50000,
        test_size=10000,
        num_classes=10,
        regime=_REGIME_ID,
        url="https://www.cs.toronto.edu/~kriz/cifar.html",
        notes="CIFAR-10.",
    ),
    "cifar100": DatasetInfo(
        name="cifar100",
        train_size=50000,
        test_size=10000,
        num_classes=100,
        regime=_REGIME_ID,
        url="https://www.cs.toronto.edu/~kriz/cifar.html",
        notes="CIFAR-100.",
    ),
    "voc2007": DatasetInfo(
        name="voc2007",
        train_size=2501,
        test_size=2510,
        num_classes=20,
        regime=_REGIME_ID,
        url="http://host.robots.ox.ac.uk/pascal/VOC/voc2007",
        notes="PASCAL VOC 2007; multi-class classification protocol.",
    ),
    "sun397": DatasetInfo(
        name="sun397",
        train_size=108754,
        test_size=None,  # No official benchmark test split used in this protocol.
        num_classes=397,
        regime=_REGIME_OOD,
        url="https://groups.csail.mit.edu/vision/SUN/hierarchy.html",
        notes=(
            "SUN397.  No official test split is used; evaluation is performed "
            "only on the training portion.  Attempting to load a test split "
            "will raise ValueError."
        ),
    ),
    "dtd": DatasetInfo(
        name="dtd",
        train_size=4512,
        test_size=1128,
        num_classes=47,
        regime=_REGIME_OOD,
        url="https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        notes="Describable Textures Dataset.",
    ),
}

# Regime groupings — used by the evaluation harness.
REGIMES: Dict[str, List[str]] = {
    _REGIME_ID: ["caltech101", "cifar10", "cifar100", "voc2007"],
    _REGIME_OOD: ["dtd", "sun397"],
    _REGIME_FG: ["aircraft", "cars", "flowers", "food", "pets"],
}


def get_dataset(name: str) -> DatasetInfo:
    """Return the :class:`DatasetInfo` record for *name*.

    Parameters
    ----------
    name:
        Dataset key (case-insensitive).

    Returns
    -------
    DatasetInfo

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    key = name.lower()
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}.  "
            f"Available: {sorted(DATASETS.keys())}"
        )
    return DATASETS[key]
