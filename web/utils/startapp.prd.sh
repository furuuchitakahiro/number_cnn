#!/bin/sh
pip install -r requirements.txt
gunicorn -b :$PORT src.app:app
