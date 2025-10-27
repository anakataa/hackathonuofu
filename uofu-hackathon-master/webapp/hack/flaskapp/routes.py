"""Flask routes and Jinja context information"""

import json
import os

from flask import abort, make_response, redirect, request
from hack import const, utils
from hack.flaskapp import app
from hack.logger import log


@app.context_processor
def context_processor() -> dict:
    # The default set of variables provided to the templating engine (Jinja2).

    # Initializing this with the dict() constructor instead of {} forces all keys to be valid identifiers
    base_context: dict = dict(
        snake_name=const.snake_name,
        pascal_name=const.pascal_name,
        pretty_name=const.pretty_name,
        logo_small=const.logo_small,
        logo_icon=const.logo_icon,
    )

    # extra_context takes precedence over conflicting fields in base_context.
    extra_context: dict = utils.get_extra_context()
    total_context: dict = base_context | extra_context

    return total_context


@app.route(const.root_page)
@app.route(const.home_page)
def home_page():
    utils.add_extra_context(topic="Mars Weather [LIVE]")

    days = json.load(open(os.path.join(const.flaskapp_dir_path, "days.json"), "r"))

    return utils.render_page(const.home_page, days=days)
