"""Entry point:  uvicorn app.server:app --reload"""
from .main import create_app

app = create_app()
