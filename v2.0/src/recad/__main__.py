"""Allow ``python -m recad`` as an alias of the ``recad`` console script."""

import sys

from recad.cli import main

if __name__ == "__main__":
    sys.exit(main())
