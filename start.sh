#!/bin/bash
apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg
gunicorn --preload -w 1 -b 0.0.0.0:5000 app:app --timeout 120
