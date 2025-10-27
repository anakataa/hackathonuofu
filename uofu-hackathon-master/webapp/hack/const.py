"""Global constants"""

import os


# Constants for convenience
empty_str: str = ""

# Abbreviated names
snake_name: str = "hackathon"  # snake_case
pascal_name: str = "Hackathon"  # Pascal Case (with spaces separating words)

# Full names
pretty_name: str = "Hackathon"

# Logos
# TODO: Consider replacing these company-specific logos with our own.
logo_small: str = "pages/common/media/example.png"
logo_icon: str = empty_str  # TODO: Find an icon for this web-app

# Root directory of this project
root_dir_path: str = os.path.dirname(__file__)
root_dir_name: str = os.path.basename(root_dir_path)

# Directory containing the Flask app
flaskapp_dir_name: str = "flaskapp"
flaskapp_dir_path: str = os.path.join(root_dir_path, flaskapp_dir_name)

# Directory containing the HTML, CSS, and JS files for the Flask app
pages_dir_name: str = "pages"
pages_dir_path: str = os.path.join(flaskapp_dir_path, pages_dir_name)

# All scripts have a Python file extension
script_ext: str = ".py"

# All pages have an index file that is rendered by Jinja before being sent to the user
index_file_name: str = "index.html"

# Log files have a common file extension
log_ext: str = ".log"

# All log files are stored in a common directory
logs_dir_name: str = "logs"
logs_dir_path: str = os.path.join(flaskapp_dir_path, logs_dir_name)

# Page routes
root_page: str = "/"  # The root page is the index page
home_page: str = "/home"
