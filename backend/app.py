# backend/app.py

from flask import Flask, jsonify, request
import os
import logging
import requests # For making HTTP requests
import base64 # For encoding image data for AI
import json # For parsing JSON responses from AI
import openai # Added for OpenAI
from io import BytesIO # Potentially useful for image handling

# Configure basic logging based on environment variable or default
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Global Variables (from Addon Configuration via Environment Variables) ---
ADDON_PORT = int(os.environ.get("PORT", 8099))
DEFAULT_CAMERA_ENTITY_ID = os.environ.get("CAMERA_ENTITY_ID", "camera.your_room_camera")
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower() # Ensure lowercase for easier comparison
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") # Used if AI_PROVIDER is 'openai'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") # Used if AI_PROVIDER is 'google_gemini'
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o") # Configurable OpenAI model

# Home Assistant Supervisor Token and API URL
HA_TOKEN = os.environ.get("HA_TOKEN")
HA_URL = os.environ.get("HA_URL", "http://supervisor/core/api")

logging.info(f"AI Provider set to: {AI_PROVIDER}")
if AI_PROVIDER == "openai":
    logging.info(f"OpenAI Model set to: {OPENAI_MODEL}")
    if not OPENAI_API_KEY:
        logging.warning("AI_PROVIDER is openai, but OPENAI_API_KEY is not set.")
elif AI_PROVIDER == "google_gemini":
    if not GEMINI_API_KEY:
        logging.warning("AI_PROVIDER is google_gemini, but GEMINI_API_KEY is not set.")

logging.info(f"HA_URL: {HA_URL}")
if not HA_TOKEN:
    logging.warning("HA_TOKEN (SUPERVISOR_TOKEN) is not set. Camera fetching will fail.")
else:
    logging.info("HA_TOKEN is available.")


# --- Mock Data (to be replaced with actual logic and database) ---
mock_spaces = {
    "kitchen": {
        "name": "Kitchen Space",
        "camera_entity_id": DEFAULT_CAMERA_ENTITY_ID, # Can be overridden per space
        "todos": [
            {"id": 1, "task": "Return cooking oil bottles to cabinet", "completed": False, "created_at": "1d ago"},
            {"id": 2, "task": "Load dirty dishes into dishwasher", "completed": False, "created_at": "1d ago"}
        ],
        "stats": {
            "pending": 2,
            "completed_today": 0,
            "streak": 0,
            "completion_percentage": 0,
            "tasks_created_total": 2,
            "tasks_completed_total": 0
        }
    },
    "living_room": {
        "name": "Living Room",
        "camera_entity_id": "camera.living_room_cam",
        "todos": [],
        "stats": {
            "pending": 0,
            "completed_today": 0,
            "streak": 0,
            "completion_percentage": 0,
            "tasks_created_total": 0,
            "tasks_completed_total": 0
        }
    }
}

# --- Helper Function to Fetch Camera Image ---
def fetch_camera_image(camera_entity_id):
    if not HA_TOKEN:
        logging.error("Cannot fetch camera image: Home Assistant token (HA_TOKEN) is not available.")
        return None
    if not camera_entity_id:
        logging.error("Cannot fetch camera image: Camera entity ID is not provided.")
        return None

    image_url = f"{HA_URL}/camera_proxy/{camera_entity_id}"
    headers = {"Authorization": f"Bearer {HA_TOKEN}"}
    logging.info(f"Fetching image from URL: {image_url}")
    try:
        response = requests.get(image_url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status()
        logging.info(f"Successfully fetched image for {camera_entity_id}. Content-Type: {response.headers.get('Content-Type')}")
        return response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching image for {camera_entity_id}: {e}")
    return None

# --- AI Analysis Function (Gemini) ---
def analyze_image_with_gemini(image_bytes, room_name): # Removed async
    """
    Analyzes an image using Google Gemini to identify cleaning tasks.
    Returns a list of task strings or None if an error occurs.
    """
    if not image_bytes:
        logging.error("Gemini analysis: No image bytes provided.")
        return None

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = (
        f"Analyze the provided image of a '{room_name}'. "
        "Identify any items that are out of place, messes, or areas that need cleaning. "
        "Based on your analysis, provide a list of specific, actionable cleaning tasks. "
        "Return these tasks as a JSON array of strings. Each string should be a concise to-do item. "
        "For example: [\"Wipe down the kitchen counter\", \"Put the scattered books back on the shelf\"]. "
        "If the room looks clean and no tasks are identified, return an empty array []."
    )

    if not GEMINI_API_KEY:
        logging.error("Gemini API key is not configured.")
        return None
    gemini_model_name = "gemini-1.5-flash-latest" 
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model_name}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg", 
                            "data": image_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "A list of cleaning tasks."
            }
        }
    }

    headers = {'Content-Type': 'application/json'}
    logging.info(f"Sending request to Gemini API for {room_name} analysis.")

    try:
        response = requests.post(gemini_url, headers=headers, json=payload, timeout=30) 
        response.raise_for_status()
        
        result = response.json()
        logging.debug(f"Gemini API raw response: {result}")

        if (result.get("candidates") and result["candidates"][0].get("content") and
                result["candidates"][0]["content"].get("parts") and
                len(result["candidates"][0]["content"]["parts"]) > 0):
            
            tasks_text = result["candidates"][0]["content"]["parts"][0].get("text")
            if tasks_text:
                try:
                    tasks_list = json.loads(tasks_text)
                    if isinstance(tasks_list, list) and all(isinstance(task, str) for task in tasks_list):
                        logging.info(f"Gemini identified {len(tasks_list)} tasks for {room_name}.")
                        return tasks_list
                    else:
                        logging.error(f"Gemini response schema mismatch: Expected list of strings, got: {tasks_text}")
                        return [] 
                except json.JSONDecodeError:
                    logging.error(f"Gemini response JSON decode error for tasks: {tasks_text}")
                    return [] 
            else:
                logging.warning(f"Gemini response missing 'text' in parts for {room_name}.")
                return []
        else:
            logging.warning(f"Gemini response structure unexpected or content missing for {room_name}. Response: {result}")
            if result.get("promptFeedback"):
                logging.warning(f"Gemini prompt feedback: {result.get('promptFeedback')}")
            return [] 
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"Gemini API HTTP error: {http_err} - Response: {response.text if 'response' in locals() else 'No response object'}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Gemini API request error: {req_err}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during Gemini API call: {e}")
    
    return None

# --- AI Analysis Function (OpenAI) ---
def analyze_image_with_openai(image_bytes, room_name): # New function
    """
    Analyzes an image using OpenAI (e.g., GPT-4o) to identify cleaning tasks.
    Returns a list of task strings or None if an error occurs.
    """
    if not image_bytes:
        logging.error("OpenAI analysis: No image bytes provided.")
        return None

    if not OPENAI_API_KEY:
        logging.error("OpenAI API key is not configured.")
        return None

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return None

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt_text = (
        f"Analyze the provided image of a '{room_name}'. "
        "Identify any items that are out of place, messes, or areas that need cleaning. "
        "Based on your analysis, provide a list of specific, actionable cleaning tasks. "
        "Return these tasks as a JSON array of strings. Each string should be a concise to-do item. "
        "For example: [\"Wipe down the kitchen counter\", \"Put the scattered books back on the shelf\"]. "
        "If the room looks clean and no tasks are identified, return an empty array []."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}", # Assuming JPEG
                        "detail": "low" # or "high" or "auto"
                    }
                }
            ]
        }
    ]

    logging.info(f"Sending request to OpenAI API ({OPENAI_MODEL}) for {room_name} analysis.")
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=500, # Adjust as needed
            temperature=0.3, # Adjust for creativity vs. directness
            # Ensure the model can output JSON, or instruct it clearly in the prompt
            # For newer models, you can use response_format={"type": "json_object"}
            # if the model supports it and the prompt is structured for it.
            # For now, relying on prompt instruction for JSON array.
        )
        
        response_content = completion.choices[0].message.content
        logging.debug(f"OpenAI API raw response content: {response_content}")

        if response_content:
            try:
                # Attempt to find a JSON array within the response, as AI might add extra text
                # A more robust way is to use response_format={"type": "json_object"} if available
                # and structure the prompt to ask for a JSON object with a specific key for the array.
                # For now, simple extraction:
                json_start = response_content.find('[')
                json_end = response_content.rfind(']') + 1
                if json_start != -1 and json_end != 0 and json_end > json_start:
                    tasks_text = response_content[json_start:json_end]
                    tasks_list = json.loads(tasks_text)
                    if isinstance(tasks_list, list) and all(isinstance(task, str) for task in tasks_list):
                        logging.info(f"OpenAI ({OPENAI_MODEL}) identified {len(tasks_list)} tasks for {room_name}.")
                        return tasks_list
                    else:
                        logging.error(f"OpenAI response schema mismatch: Expected list of strings, got: {tasks_text}")
                        return []
                else: # If no clear JSON array found, try parsing the whole thing
                    tasks_list = json.loads(response_content) # This might fail if not pure JSON
                    if isinstance(tasks_list, list) and all(isinstance(task, str) for task in tasks_list):
                         logging.info(f"OpenAI ({OPENAI_MODEL}) identified {len(tasks_list)} tasks for {room_name} (parsed whole response).")
                         return tasks_list
                    else:
                        logging.error(f"OpenAI response schema mismatch (parsed whole response): Expected list of strings, got: {response_content}")
                        return []

            except json.JSONDecodeError:
                logging.error(f"OpenAI response JSON decode error: {response_content}")
                # Fallback: try to extract tasks if it's a simple list not in JSON array format
                # This is a very basic fallback and might not be robust.
                if "\n-" in response_content or "\n*" in response_content:
                    potential_tasks = [line.strip('-* ') for line in response_content.split('\n') if line.strip('-* ')]
                    if potential_tasks:
                        logging.info(f"OpenAI response was not JSON, but extracted {len(potential_tasks)} potential tasks from list format.")
                        return potential_tasks
                return [] # Return empty list if JSON parsing fails and no fallback
        else:
            logging.warning(f"OpenAI response content empty for {room_name}.")
            return []

    except openai.APIError as api_err:
        logging.error(f"OpenAI API error: {api_err} - Status: {api_err.status_code} - Response: {api_err.response}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during OpenAI API call: {e}")

    return None # Return None on error


# --- API Endpoints ---
@app.route('/')
def index():
    logging.info("Root endpoint '/' was accessed.")
    return jsonify({
        "message": "AI Room Cleaner Addon Backend is running!",
        "version": "0.1.2", # Incremented version
        "ha_token_available": bool(HA_TOKEN),
        "ai_provider": AI_PROVIDER,
        "openai_model_configured": OPENAI_MODEL if AI_PROVIDER == "openai" else "N/A"
    })

@app.route('/api/spaces', methods=['GET'])
def get_spaces():
    logging.info("GET /api/spaces was accessed.")
    spaces_overview = []
    for space_id, space_data in mock_spaces.items():
        spaces_overview.append({
            "id": space_id,
            "name": space_data["name"],
            "pending_todos": space_data["stats"]["pending"],
            "completion_percentage": space_data["stats"]["completion_percentage"]
        })
    return jsonify(spaces_overview)

@app.route('/api/spaces/<space_id>', methods=['GET'])
def get_space_details(space_id):
    logging.info(f"GET /api/spaces/{space_id} was accessed.")
    space = mock_spaces.get(space_id.lower())
    if not space:
        logging.warning(f"Space '{space_id}' not found.")
        return jsonify({"error": "Space not found"}), 404
    return jsonify(space)

@app.route('/api/spaces/<space_id>/todos', methods=['GET'])
def get_todos(space_id):
    logging.info(f"GET /api/spaces/{space_id}/todos was accessed.")
    space = mock_spaces.get(space_id.lower())
    if not space:
        logging.warning(f"Space '{space_id}' not found.")
        return jsonify({"error": "Space not found"}), 404
    return jsonify(space["todos"])

@app.route('/api/spaces/<space_id>/todos', methods=['POST'])
def add_todo(space_id):
    logging.info(f"POST /api/spaces/{space_id}/todos was accessed.")
    space = mock_spaces.get(space_id.lower())
    if not space:
        logging.warning(f"Space '{space_id}' not found for adding todo.")
        return jsonify({"error": "Space not found"}), 404

    if not request.json or 'task' not in request.json:
        logging.warning("Task not provided in request for adding todo.")
        return jsonify({"error": "Task not provided"}), 400

    new_task_text = request.json['task']
    new_todo_id = max([t["id"] for t in space["todos"]] + [0]) + 1
    new_todo = {
        "id": new_todo_id,
        "task": new_task_text,
        "completed": False,
        "created_at": "just now (manual)" 
    }
    space["todos"].append(new_todo)
    space["stats"]["pending"] += 1
    space["stats"]["tasks_created_total"] += 1
    total_tasks = len(space["todos"])
    completed_tasks = sum(1 for t in space["todos"] if t["completed"])
    space["stats"]["completion_percentage"] = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    logging.info(f"Manually added new todo to space '{space_id}': {new_task_text}")
    return jsonify(new_todo), 201


@app.route('/api/spaces/<space_id>/todos/<int:todo_id>', methods=['PUT'])
def update_todo(space_id, todo_id):
    logging.info(f"PUT /api/spaces/{space_id}/todos/{todo_id} was accessed.")
    space = mock_spaces.get(space_id.lower())
    if not space:
        logging.warning(f"Space '{space_id}' not found for updating todo.")
        return jsonify({"error": "Space not found"}), 404

    todo_to_update = next((t for t in space["todos"] if t["id"] == todo_id), None)
    if not todo_to_update:
        logging.warning(f"Todo ID '{todo_id}' not found in space '{space_id}'.")
        return jsonify({"error": "Todo not found"}), 404

    update_data = request.json
    changed = False
    if 'completed' in update_data:
        was_completed = todo_to_update["completed"]
        is_now_completed = bool(update_data['completed'])
        if was_completed != is_now_completed:
            todo_to_update["completed"] = is_now_completed
            changed = True
            if is_now_completed: 
                space["stats"]["pending"] = max(0, space["stats"]["pending"] - 1)
                space["stats"]["completed_today"] += 1
                space["stats"]["tasks_completed_total"] += 1
                space["stats"]["streak"] +=1
                logging.info(f"Todo ID '{todo_id}' in space '{space_id}' marked as complete.")
            else: 
                space["stats"]["pending"] += 1
                space["stats"]["completed_today"] = max(0, space["stats"]["completed_today"] - 1)
                space["stats"]["tasks_completed_total"] = max(0, space["stats"]["tasks_completed_total"] -1)
                space["stats"]["streak"] = max(0, space["stats"]["streak"] -1)
                logging.info(f"Todo ID '{todo_id}' in space '{space_id}' marked as incomplete.")

    if 'task' in update_data and todo_to_update['task'] != update_data['task']:
        todo_to_update['task'] = update_data['task']
        changed = True
        logging.info(f"Todo ID '{todo_id}' in space '{space_id}' text updated.")

    if changed:
        total_tasks = len(space["todos"])
        completed_tasks = sum(1 for t in space["todos"] if t["completed"])
        space["stats"]["completion_percentage"] = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    return jsonify(todo_to_update)

@app.route('/api/spaces/<space_id>/stats', methods=['GET'])
def get_stats(space_id):
    logging.info(f"GET /api/spaces/{space_id}/stats was accessed.")
    space = mock_spaces.get(space_id.lower())
    if not space:
        logging.warning(f"Space '{space_id}' not found for fetching stats.")
        return jsonify({"error": "Space not found"}), 404
    return jsonify(space["stats"])

@app.route('/api/analyze-room/<space_id>', methods=['POST'])
def analyze_room_endpoint(space_id): # Removed async
    logging.info(f"POST /api/analyze-room/{space_id} was accessed.")
    space = mock_spaces.get(space_id.lower())
    if not space:
        logging.warning(f"Space '{space_id}' not found for analysis.")
        return jsonify({"error": "Space not found"}), 404

    room_name = space.get("name", space_id) 
    camera_entity_to_use = space.get("camera_entity_id", DEFAULT_CAMERA_ENTITY_ID)
    logging.info(f"Attempting to use camera: {camera_entity_to_use} for space {space_id} ('{room_name}')")

    image_bytes = fetch_camera_image(camera_entity_to_use)
    if not image_bytes:
        logging.error(f"Failed to fetch image for {camera_entity_to_use}. Cannot proceed with AI analysis.")
        return jsonify({"error": f"Failed to fetch image from camera {camera_entity_to_use}"}), 500
    
    logging.info(f"Successfully fetched image for {camera_entity_to_use} ({len(image_bytes)} bytes). AI Provider: {AI_PROVIDER}")

    ai_generated_tasks = []
    analysis_provider_name = ""

    if AI_PROVIDER == "google_gemini":
        analysis_provider_name = "Gemini"
        tasks_from_ai = analyze_image_with_gemini(image_bytes, room_name) # Removed await
        if tasks_from_ai is not None: 
            ai_generated_tasks = tasks_from_ai
            logging.info(f"Gemini AI analysis complete for {room_name}. Tasks found: {len(ai_generated_tasks)}")
        else:
            logging.error(f"Gemini AI analysis failed for {room_name}.")
            return jsonify({"error": f"AI analysis failed for {room_name} using {analysis_provider_name}"}), 500
            
    elif AI_PROVIDER == "openai":
        analysis_provider_name = f"OpenAI ({OPENAI_MODEL})"
        tasks_from_ai = analyze_image_with_openai(image_bytes, room_name) # New call
        if tasks_from_ai is not None:
            ai_generated_tasks = tasks_from_ai
            logging.info(f"OpenAI ({OPENAI_MODEL}) AI analysis complete for {room_name}. Tasks found: {len(ai_generated_tasks)}")
        else:
            logging.error(f"OpenAI ({OPENAI_MODEL}) AI analysis failed for {room_name}.")
            return jsonify({"error": f"AI analysis failed for {room_name} using {analysis_provider_name}"}), 500
    else:
        logging.warning(f"Unsupported AI provider: {AI_PROVIDER}. No analysis performed.")
        return jsonify({"error": f"Unsupported AI provider: {AI_PROVIDER}"}), 400

    newly_added_tasks_count = 0
    if ai_generated_tasks: 
        existing_pending_tasks_text = [t["task"] for t in space["todos"] if not t["completed"]]
        for task_text in ai_generated_tasks:
            if task_text.lower() not in (existing_task.lower() for existing_task in existing_pending_tasks_text):
                new_todo_id = max([t["id"] for t in space["todos"]] + [0]) + 1
                new_todo = {
                    "id": new_todo_id,
                    "task": task_text, 
                    "completed": False,
                    "created_at": f"just now ({analysis_provider_name} AI)"
                }
                space["todos"].append(new_todo)
                space["stats"]["pending"] += 1
                space["stats"]["tasks_created_total"] += 1
                newly_added_tasks_count += 1
                logging.info(f"AI ({analysis_provider_name}) generated and added new task to '{space_id}': {task_text}")
            else:
                logging.info(f"AI ({analysis_provider_name}) generated task '{task_text}' for '{space_id}' is a duplicate of a pending task. Skipping.")

    total_tasks = len(space["todos"])
    completed_tasks = sum(1 for t in space["todos"] if t["completed"])
    space["stats"]["completion_percentage"] = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    
    if newly_added_tasks_count > 0:
         return jsonify({
            "message": f"{analysis_provider_name} AI analysis complete for {room_name}. {newly_added_tasks_count} new task(s) added.",
            "tasks_identified": ai_generated_tasks, 
            "newly_added_count": newly_added_tasks_count
        }), 200
    elif ai_generated_tasks is not None and not ai_generated_tasks: 
        return jsonify({
            "message": f"{analysis_provider_name} AI analysis complete for {room_name}. Room appears to be clean, no new tasks added.",
            "tasks_identified": [],
            "newly_added_count": 0
        }), 200
    else: 
         return jsonify({
            "message": f"{analysis_provider_name} AI analysis finished for {room_name}. No new distinct tasks added.",
            "tasks_identified": ai_generated_tasks if ai_generated_tasks is not None else "Analysis might have failed or provider not fully supported",
            "newly_added_count": 0
        }), 200

# --- Main Execution ---
if __name__ == '__main__':
    logging.info(f"Starting Flask backend server on host 0.0.0.0 port {ADDON_PORT}...")
    app.run(host='0.0.0.0', port=ADDON_PORT, debug=True)
