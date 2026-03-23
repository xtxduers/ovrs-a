# Copyright (c) Facebook, Inc. and its affiliates.
"""
Register all custom datasets with Detectron2's DatasetCatalog / MetadataCatalog.

Each sub-module registers its datasets at import time by calling
``DatasetCatalog.register()``.  The ``DETECTRON2_DATASETS`` environment
variable controls the root directory that is searched for dataset files
(default: ``"datasets"``).
"""

# Aerial / satellite datasets (benchmarks from the original CATSeg paper)
from . import register_isaid      # noqa: F401
from . import register_dlrsd      # noqa: F401
from . import register_potsdam    # noqa: F401
from . import register_vaihingen  # noqa: F401

# Drone / UAV datasets
from . import register_loveda     # noqa: F401
from . import register_uavid      # noqa: F401
from . import register_udd5       # noqa: F401
from . import register_vdd        # noqa: F401
