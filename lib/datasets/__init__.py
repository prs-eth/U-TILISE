from typing import Dict, Type

from .EarthNet2021Dataset import EarthNet2021Dataset
from .SEN12MSCRTSDataset import SEN12MSCRTSDataset

DATASETS: Dict[str, Type[EarthNet2021Dataset] | Type[SEN12MSCRTSDataset]] = {
    "earthnet2021": EarthNet2021Dataset,
    "sen12mscrts": SEN12MSCRTSDataset
}
