from enum import Enum


class ActivationType(Enum):
    RELU = 'relu'
    LRELU = 'lrelu'
    PRELU = 'prelu'
    MISH = 'mish'
    GELU = 'gelu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'


class LTAENormType(Enum):
    GROUP = 'group'
    LAYER = 'layer'


class NormType(Enum):
    BATCH = 'batch'
    GROUP = 'group'
    INSTANCE = 'instance'
    NONE = None


class UpConvType(Enum):
    TRANSPOSE = 'transpose'
    BILINEAR = 'bilinear'


class TemporalAggregationMode(Enum):
    ATT_GROUP = 'att_group'
    ATT_MEAN = 'att_mean'
    NONE = None
