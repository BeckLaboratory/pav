"""Low-confidence (LC) alignment prediction.

LC models examine alignment features and make a confidence prediction. The built-in logistic model,
:class:`LCAlignModelLogistic`, uses logistic regression on a list af alignment features. The built-in null model,
:class:LCAlignModelNull`, does not predict any LC alignments.
"""

# TODO: Add feature scaling

__all__ = [
    'LCAlignModel',
    'LCAlignModelLogistic',
    'LCAlignModelNull',
    'get_model',
    'locate_config_filesystem',
    'locate_config_package',
    'locate_model',
    'null_model',
]

from ._lcmodel import LCAlignModel
from ._lcmodel_logistic import LCAlignModelLogistic
from ._lcmodel_null import LCAlignModelNull
from ._util import (
    get_model,
    locate_config_filesystem,
    locate_config_package,
    locate_model,
    null_model
)
