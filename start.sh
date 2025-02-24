#!/bin/bash 
# Install FFmpeg
apt-get update && apt-get install -y ffmpeg

# Set execution permission
chmod +x start.sh  

# Start the Flask app using Gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 app:app --timeout 120
