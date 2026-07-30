import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'lifelink-secret-key-2024'
    DEBUG = True

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATABASE = os.path.join(BASE_DIR, 'lifelink.db')