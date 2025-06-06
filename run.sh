#!/usr/bin/with-contenv bashio
# run.sh
# This script is executed when the addon starts.

# Bashio is a Home Assistant utility library for bash scripts.
# It provides functions to read config options, log messages, etc.
# More info: https://developers.home-assistant.io/docs/add-ons/bashio

bashio::log.info "Starting AI Room Cleaner Addon..."

# Read configuration options from /data/options.json (populated by Home Assistant)
CONFIG_PATH=/data/options.json

LOG_LEVEL=$(bashio::config 'log_level' 'info')
CAMERA_ENTITY=$(bashio::config 'default_camera_entity' 'camera.your_room_camera')
AI_PROVIDER=$(bashio::config 'ai_provider' 'openai')
OPENAI_API_KEY=$(bashio::config 'openai_api_key' '')
GEMINI_API_KEY=$(bashio::config 'gemini_api_key' '')
# Add other config variables as needed

# Export environment variables that your Python app will use
export PYTHONUNBUFFERED=1 # Ensures Python logs are not buffered
export LOG_LEVEL="${LOG_LEVEL}"
export CAMERA_ENTITY_ID="${CAMERA_ENTITY}" # Note: app.py uses CAMERA_ENTITY_ID
export AI_PROVIDER="${AI_PROVIDER}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export GEMINI_API_KEY="${GEMINI_API_KEY}"
# The SUPERVISOR_TOKEN is automatically available if homeassistant_api: true in config.yaml
export HA_TOKEN="${SUPERVISOR_TOKEN}" # Pass supervisor token to app.py
export HA_URL="http://supervisor/core/api" # Internal URL to HA API, app.py uses this

bashio::log.info "Log Level: ${LOG_LEVEL}"
bashio::log.info "Default Camera Entity: ${CAMERA_ENTITY}"
bashio::log.info "AI Provider: ${AI_PROVIDER}"
if [[ "${AI_PROVIDER}" == "openai" ]]; then
    if [ -n "${OPENAI_API_KEY}" ]; then
        bashio::log.info "OpenAI API Key is set."
    else
        bashio::log.warning "OpenAI API Key is NOT set. OpenAI features will not work."
    fi
elif [[ "${AI_PROVIDER}" == "google_gemini" ]]; then
    if [ -n "${GEMINI_API_KEY}" ]; then
        bashio::log.info "Google Gemini API Key is set."
    else
        bashio::log.warning "Google Gemini API Key is NOT set. Gemini features will not work."
    fi
fi
if [ -n "${SUPERVISOR_TOKEN}" ]; then
    bashio::log.info "Home Assistant Supervisor token is available."
else
    bashio::log.warning "Home Assistant Supervisor token is NOT available. Cannot fetch camera images."
fi


# Navigate to the app directory (where app.py is)
cd /app || exit 1

bashio::log.info "Starting Python Flask backend server..."

# Execute the Python application
# Use exec to replace the shell process with the Python process
exec python3 ./app.py
# If you were using gunicorn for production:
# exec gunicorn --workers 4 --bind 0.0.0.0:8099 app:app

bashio::log.info "AI Room Cleaner Addon has stopped." # This line might not be reached if exec is used