from importlib.metadata import version, PackageNotFoundError

__all__ = ['fit', 'utils', "__version__"]

from .fit import *
from .utils import *


try:
    # Get the version of the installed 'astra' package
    __version__ = version("astra")
except PackageNotFoundError:
    # Fallback if the package is not installed (e.g., running from source)
    # You might want to log a warning here or set a default
    raise RuntimeError(