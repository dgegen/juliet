from importlib.metadata import PackageNotFoundError, version

__all__ = ["fit", "load", "model", "gaussian_process", "utils", "__version__"]

import juliet.utils as utils

from .fit import fit
from .gaussian_process import gaussian_process
from .load import load
from .model import model

try:
    # Get the version of the installed 'astra' package
    __version__ = version("juliet")
except PackageNotFoundError:
    raise RuntimeError(
        "Package 'juliet' is not installed. "
        "Please install it to use the package functionalities."
    )
