#!/bin/bash
# Install FFmpeg
apt-get update && apt-get install -y ffmpeg
# Start the Flask app using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
