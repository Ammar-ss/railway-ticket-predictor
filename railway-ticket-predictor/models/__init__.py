from flask import Flask

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)

    with app.app_context():
        from . import routes

    return app
