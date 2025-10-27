# **Mars Weather Prediction**

An interactive web-app for predicting the weather on Mars. 

## Quickstart

#### 1. Install required packages.

Install NPM and Python.

##### Linux (Arch):

```bash
sudo pacman -S python npm
```

##### MacOS:

```zsh
brew install python npm
```

#### 2. Clone this project and enter its root directory.

```bash
git clone https://github.com/cshmookler/uofu-hackathon
cd uofu-hackathon/webapp
```

#### 3. Install dependencies for Python.

Create a Python virtual environment so dependencies are installed locally instead of globally.

```bash
python3 -m venv .venv
```

Activate the virtual environment. Any calls to 'python3' or 'pip3' will reference the executables in the (local) virtual environment instead of your global environment.

> NOTE: The virtual environment must be reactivated when starting the server in a new environment.

```bash
source .venv/bin/activate
```

Install all required dependencies in the virtual environment.

```bash
pip3 install -r python_requirements.txt
```

#### 4. Install a local copy of Bootstrap-5.3.3+ and a SASS compiler.

```bash
npm install bootstrap sass
```

#### 5. Compile the custom Bootstrap styles.

```bash
BS_OVERRIDE='hack/flaskapp/pages/common/css/custom_bootstrap'
```

To compile once:

```bash
./node_modules/sass/sass.js "$BS_OVERRIDE.scss" "$BS_OVERRIDE.css"
```

Add the `--watch` flag to recompile whenever the override file is updated.

```bash
./node_modules/sass/sass.js --watch "$BS_OVERRIDE.scss" "$BS_OVERRIDE.css"
```

#### 6. Start the server.

###### Development

```bash
flask --app config.py run --debug --reload --host 0.0.0.0 --port 5000
```

###### Production

```bash
gunicorn --config config.py
```
