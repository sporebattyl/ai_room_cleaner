# Dockerfile
# Defines how to build the Docker image for the addon.

# Use an official Home Assistant base image with Python pre-installed.
ARG BUILD_FROM="homeassistant/amd64-base-python:3.11" # Default if not overridden by build.yaml
FROM ${BUILD_FROM}

# Set environment variables (can be overridden by config.yaml or run.sh)
ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED 1 # Ensures print statements and logs appear immediately

# Install system dependencies if needed (e.g., for image processing libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ffmpeg \
#  && rm -rf /var/lib/apt/lists/*

# Create a working directory for the addon
WORKDIR /app

# Copy the backend application code into the image
# Assuming your app.py and any requirements.txt are in a 'backend' subdirectory
COPY backend/requirements.txt .
COPY backend/app.py .
# If you have other files/folders in 'backend', copy them too:
# COPY backend/static ./static
# COPY backend/templates ./templates

# Install Python dependencies
# It's good practice to use a requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the Flask app will run on (must match config.yaml and app.py)
EXPOSE 8099

# Copy the run script and make it executable
COPY run.sh /
RUN chmod +x /run.sh

# Command to run when the container starts (defined in run.sh)
CMD [ "/run.sh" ]