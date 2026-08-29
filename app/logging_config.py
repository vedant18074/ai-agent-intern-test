"""Logging configuration for the Aster & Row support agent.

The main agent currently configures logging itself so this module is optional.
It is provided as a small reusable configuration helper for future modules.
"""

import logging
import os


def setup_logging() -> None:
    level = (
        logging.DEBUG
        if os.getenv("DEBUG", "false").lower() == "true"
        else logging.INFO
    )

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )
