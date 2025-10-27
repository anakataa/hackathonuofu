import flask
from hack import const


# Initialize Flask
app = flask.Flask(
    const.pretty_name,
    root_path=const.flaskapp_dir_path,
    static_folder=const.pages_dir_name,
    template_folder=const.pages_dir_name,
)

del flask
del const

# Add all routes
from hack.flaskapp import routes
