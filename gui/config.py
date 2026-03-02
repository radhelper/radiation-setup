import os

class FlaskConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-might-actually-guess'