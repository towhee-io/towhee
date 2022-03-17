import importlib

import importlib_metadata
from towhee.utils.log import trainer_log

_captum_available = importlib.util.find_spec("captum") is not None
try:
    _captum_version = importlib_metadata.version("captum")
    trainer_log.info("Successfully imported captum version %s", _captum_version)
except importlib_metadata.PackageNotFoundError:
    _captum_available = False

_tensorboard_available = importlib.util.find_spec("tensorboard") is not None \
                         or importlib.util.find_spec("tensorboardX") is not None
try:
    from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-outside-toplevel
except Exception:
    _tensorboard_available = False


def is_captum_available():
    return _captum_available


def is_tensorboard_available():
    return _tensorboard_available


def is_matplotlib_available():
    return importlib.util.find_spec("matplotlib") is not None
