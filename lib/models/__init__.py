from typing import Dict, Type

from .ImageSeriesInterpolator import ImageSeriesInterpolator
from .utilise import UTILISE

MODELS: Dict[str, Type[UTILISE | ImageSeriesInterpolator]] = {
    "utilise": UTILISE,
    "ImageSeriesInterpolator": ImageSeriesInterpolator
}
