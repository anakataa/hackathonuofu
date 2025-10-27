"""Miscellaneous utility functions"""

import datetime
import os
import time
from types import ModuleType
from typing import Any, Optional

from flask import (  # Reset after each request.
    g as request_metadata,
    make_response,
    render_template,
    Response,
)
from hack import const


# General Utilities
# -----------------------------------------------------------------------------
def render_page(route: str, **extra_context: Any) -> Response:
    """Render and return the index page of the given route."""

    return make_response(
        render_template(f"{route}/{const.index_file_name}", **extra_context)
    )


def add_extra_context(**context: Any) -> None:
    """Add global templating context for this request."""

    # 'flask.g' is reset after each request.

    if "extra_context" not in request_metadata:
        request_metadata.extra_context = {}
    request_metadata.extra_context = request_metadata.extra_context | context


def get_extra_context() -> dict:
    """Get the global templating context for this request."""

    # 'flask.g' is reset after each request.

    if "extra_context" not in request_metadata:
        return {}
    return request_metadata.extra_context


def has_fields(map: dict, fields: dict[str, type]) -> bool:
    """Verify that a dictionary has all of the specified fields.

    Args:
        map (dict): The dictionary to check for the given fields.
        fields (dict[str, type]): {"key": type} for each field.

    Returns:
        bool: True if the dictionary has all of the specified fields and False otherwise.
    """

    for key, value_type in fields.items():
        if key not in map or type(map[key]) is not value_type:
            return False

    return True


def module_has_attrs(module: ModuleType, attrs: dict[str, type]) -> bool:
    """Verify that a module has all of the specified attributes.

    Args:
        module (ModuleType): The module to check for the given attributes.
        attrs (dict[str, type]): {"name": type} for each attribute.

    Returns:
        bool: True if the module has all of the specified attributes and False otherwise.
    """

    for attr, attr_type in attrs.items():
        if not hasattr(module, attr) or type(getattr(module, attr)) is not attr_type:
            return False

    return True


def epoch_to_date(epoch_time: int, format: str) -> str:
    """Convert a UNIX timestamp (time since Epoch) to a human-readable date.

    Args:
        epoch_time (int): The UNIX timestamp to convert to.
        format (str): The format of the human-readable date.
                      Example: "%H:%M, %Y-%m-%d"

    Returns:
        str: A human-readable date corresponding to the given timestamp and format.
    """

    return datetime.datetime.fromtimestamp(epoch_time).strftime(format)


def epoch_time() -> int:
    """Get the number of seconds since Epoch (Unix timestamp)."""

    return int(time.time())
