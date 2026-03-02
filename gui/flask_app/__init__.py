from flask import Flask

# this config import looks a bit bad
# maybe move config.py somewhere else?
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

from gui.flask_app import routes