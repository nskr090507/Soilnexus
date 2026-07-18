import os
from dotenv import load_dotenv
load_dotenv()
import json
import io
import logging
import sys
import warnings
import time
from datetime import datetime
import requests

# Suppress TensorFlow and Abseil startup logs before importing tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import firebase_admin
from firebase_admin import credentials, db
import threading
import sqlite3

def init_db():
    conn = sqlite3.connect(os.path.join(os.environ.get('TEMP', '.'), 'sensor_history.db'))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (timestamp INTEGER, moisture REAL, temperature REAL, humidity REAL, ph REAL)''')
    conn.commit()
    conn.close()

init_db()

try:
    stderr_capture = io.StringIO()
    saved_stderr = sys.stderr
    sys.stderr = stderr_capture
    try:
        import tensorflow as tf
        from PIL import Image
    finally:
        sys.stderr = saved_stderr
    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
except Exception as exc:
    tf = None
    Image = None
    print(f"TensorFlow/Pillow import failed: {exc}")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*Skipping variable loading for optimizer.*")
warnings.filterwarnings("ignore", message=".*oneDNN custom operations are on.*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=".", static_url_path="")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# The exact crop mapping used to train the machine learning model in clean_data.py
CROP_MAPPING = {
    "wheat": 0,
    "rice": 1, "paddydhancommon": 1,
    "barley": 2,
    "oats": 3,
    "maize": 4, "corn": 4,
    "sorghum": 5, "jowarsorghum": 5,
    "pearlmillet": 6, "fingermillet": 6, 
    "bajra": 7,
    "cotton": 8,
    "jute": 9,
    "sugarcane": 10,
    "tea": 11,
    "coffee": 12, "coffeerobusta": 12,
    "rubber": 13,
    "spices": 14, "garlic": 14, "ginger": 14, "cardamom": 14, "blackpepper": 14, 
    "nutmeg": 14, "cloves": 14, "cumin": 14, "fennel": 14, "coriander": 14, 
    "turmeric": 14, "fenugreek": 14, "mustard": 14, "sesame": 14, "chilli": 14,
    "fruits": 15, "apple": 15, "orange": 15, "mango": 15, "papaya": 15, "grape": 15, 
    "pomegranate": 15, "banana": 15, "coconut": 15,
    "potato": 15, "onion": 15, "tomato": 15, 
    "groundnut": 15, "peanut": 15, "cashew": 15, "almond": 15, "walnut": 15,
    "soybean": 4, "soyabean": 4, "bengalgramgramwhole": 14, "chickpea": 14,
    "tobacco": 9
}

CRED_FILENAME = os.path.join(BASE_DIR, "farmsense-ai-a40c6-firebase-adminsdk-fbsvc-3128cf3f4e.json")
FIREBASE_URL = "https://farmsense-ai-a40c6-default-rtdb.asia-southeast1.firebasedatabase.app/"

firebase_app = None
market_model = None
scaler = None
irrigation_model = None

disease_model = None
class_names = {}
label_lookup = {}
MOCK_SENSORS = None

try:
    # Railway/Cloud deployment: credentials stored as environment variable
    _firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    if _firebase_creds_json:
        _cred_dict = json.loads(_firebase_creds_json)
        cred = credentials.Certificate(_cred_dict)
        firebase_app = firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
        print("Firebase initialized from environment variable.")
    elif os.path.exists(CRED_FILENAME):
        # Local development: use credentials file
        cred = credentials.Certificate(CRED_FILENAME)
        firebase_app = firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
        print("Firebase initialized from credentials file.")
    else:
        print("Firebase credentials not found; running in demo mode.")
except Exception as exc:
    print(f"Firebase initialization skipped: {exc}")

try:
    market_model = joblib.load(os.path.join(BASE_DIR, "market_predictor.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
except Exception as exc:
    print(f"Market model loading failed: {exc}")

try:
    irrigation_model = joblib.load(os.path.join(BASE_DIR, "irrigation_model.pkl"))
except Exception as exc:
    print(f"Irrigation model loading failed: {exc}")

try:
    disease_model_path = os.path.join(BASE_DIR, "ml_model", "Plant_Disease_Detection_Model.keras")
    class_names_path = os.path.join(BASE_DIR, "ml_model", "class_names.json")

    if tf is not None and Image is not None and os.path.exists(disease_model_path):
        disease_model = tf.keras.models.load_model(disease_model_path)
    if os.path.exists(class_names_path):
        with open(class_names_path, "r", encoding="utf-8") as fp:
            class_names = json.load(fp)
            label_lookup = {int(value): key for key, value in class_names.items()}
except Exception as exc:
    print(f"Disease model initialization failed: {exc}")


def get_db_value(path, default=None):
    if firebase_app is None:
        return default
    try:
        return db.reference(path).get() or default
    except Exception as e:
        print(f"Firebase DB error for {path}: {e}")
        return default


def get_sensor_payload():
    global MOCK_SENSORS
    if MOCK_SENSORS is not None:
        sensors = MOCK_SENSORS
    else:
        sensors = get_db_value("sensor_data", {"ph": 6.5, "moisture": 50, "temperature": 25.0, "humidity": 65})
    if not isinstance(sensors, dict):
        sensors = {}
    
    # Add dynamic simulation for demo mode (no Firebase)
    if firebase_app is None and MOCK_SENSORS is None:
        import random
        current_time = time.time()
        # Simulate natural sensor fluctuations
        sensors["ph"] = max(5.5, min(7.5, sensors.get("ph", 6.5) + random.uniform(-0.1, 0.1)))
        sensors["moisture"] = max(30, min(80, sensors.get("moisture", 50) + random.uniform(-2, 2)))
        sensors["temperature"] = max(20, min(35, sensors.get("temperature", 25.0) + random.uniform(-0.5, 0.5)))
        sensors["humidity"] = max(40, min(85, sensors.get("humidity", 65) + random.uniform(-1, 1)))
        sensors["timestamp_ms"] = int(current_time * 1000)

    current_time = time.time()
    current_time_ms = int(current_time * 1000)

    # pyrefly: ignore [unknown-name]
    global last_hardware_timestamp_ms, last_hardware_receive_time
    if 'last_hardware_timestamp_ms' not in globals():
        last_hardware_timestamp_ms = None
        last_hardware_receive_time = current_time - 999999

    if isinstance(sensors, dict) and "timestamp_ms" in sensors:
        ts = sensors.get("timestamp_ms")
        if last_hardware_timestamp_ms is None:
            last_hardware_timestamp_ms = ts
        elif ts != last_hardware_timestamp_ms:
            last_hardware_timestamp_ms = ts
            last_hardware_receive_time = current_time

    last_seen_seconds = current_time - last_hardware_receive_time

    # Determine online status:
    # - Demo/simulation mode: always online
    # - Firebase connected AND sensors dict has real sensor values: treat as online
    #   (sensors are freshly read from Firebase on every request)
    # - Otherwise use timestamp staleness check
    if MOCK_SENSORS is not None or firebase_app is None:
        is_online = True
        last_seen_time_str = "Live (Demo)"
    elif isinstance(sensors, dict) and (sensors.get("moisture") is not None or sensors.get("temperature") is not None):
        # Firebase returned real data — hardware is connected
        is_online = True
        import datetime
        last_seen_time_str = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
    else:
        is_online = last_seen_seconds <= 4
        import datetime
        last_seen_time_str = (
            datetime.datetime.fromtimestamp(last_hardware_receive_time).strftime('%d/%m/%Y %H:%M')
            if last_hardware_receive_time > (current_time - 900000) else "Never"
        )

    security_status = get_db_value("security/status", "ONLINE" if is_online else "OFFLINE")

    # Read ph from sensors dict (fall back to 6.5 only if missing)
    # Clamp to a physically meaningful range (0-14)
    ph_raw = float(sensors.get("ph", sensors.get("pH", sensors.get("soil_ph", 6.5))))
    ph_val = max(0.0, min(14.0, ph_raw))
    # Dummy pH override: if sensor is broken (value is 0 or > 13), use 6.5
    if ph_val <= 0.5 or ph_val >= 13.0:
        ph_val = 6.5

    moisture_val = max(0.0, min(100.0, float(sensors.get("moisture", sensors.get("soil_moisture", 50)))))
    temperature_val = max(-20.0, min(60.0, float(sensors.get("temperature", 25.0))))
    humidity_val = max(0.0, min(100.0, float(sensors.get("humidity", sensors.get("humidity_level", 65)))))

    return {
        "ph": ph_val,
        "pH": ph_val,
        "moisture": moisture_val,
        "temperature": temperature_val,
        "humidity": humidity_val,
        "nitrogen": float(sensors.get("nitrogen", 45)),
        "phosphorus": float(sensors.get("phosphorus", 30)),
        "potassium": float(sensors.get("potassium", 120)),
        "timestamp": current_time_ms,
        "last_seen": last_seen_seconds,
        "last_seen_time_str": last_seen_time_str,
        "is_online": is_online,
        "security_status": security_status,
        "demo_mode": firebase_app is None
    }

import datetime as dt
from functools import lru_cache

@lru_cache(maxsize=128)
def get_weather_forecast(lat=28.61, lon=77.20, days=7):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum,temperature_2m_max,temperature_2m_min&timezone=auto&forecast_days={min(16, days)}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return response.json().get("daily", {})
    except Exception as e:
        print(f"Weather API error: {e}")
    return {}



MOCK_WEATHER = None

def get_default_weather():
    months_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    current_month_idx = datetime.now().month - 1
    base_temps = [15.0, 19.0, 26.0, 32.0, 35.0, 33.0, 30.0, 29.0, 28.0, 26.0, 20.0, 16.0]
    base_rains = [10.0, 15.0, 12.0, 15.0, 25.0, 70.0, 190.0, 180.0, 120.0, 35.0, 10.0, 8.0]
    
    out = []
    for i in range(12):
        m_idx = (current_month_idx + i) % 12
        out.append({
            "month": months_names[m_idx],
            "month_num": m_idx + 1,
            "temp": base_temps[m_idx],
            "rain": base_rains[m_idx]
        })
    return out


@lru_cache(maxsize=128)
def get_climate_trend(lat=28.61, lon=77.20, days=90):
    global MOCK_WEATHER
    import math
    if MOCK_WEATHER is not None:
        months_needed = math.ceil(days / 30.0)
        sliced_weather = MOCK_WEATHER[:int(months_needed)]
        if sliced_weather:
            avg_temp = sum(item["temp"] for item in sliced_weather) / len(sliced_weather)
            total_rain = sum(item["rain"] for item in sliced_weather)
            return {
                "avg_temp": avg_temp,
                "total_rain": total_rain,
                "days": days
            }

    now = dt.datetime.now()
    start_date = dt.date(now.year - 1, now.month, now.day)
    end_date = start_date + dt.timedelta(days=days)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=precipitation_sum,temperature_2m_mean&timezone=auto"
    try:
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            daily = response.json().get("daily", {})
            temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
            rains = [r for r in daily.get("precipitation_sum", []) if r is not None]
            return {
                "avg_temp": sum(temps) / len(temps) if temps else 25.0,
                "total_rain": sum(rains) if rains else 0.0,
                "days": days
            }
    except Exception as e:
        print(f"Climate API error: {e}")
    return {"avg_temp": 25.0, "total_rain": 0.0, "days": days}


def get_current_advice(crop_data, sowing_date_str):
    try:
        sowing_date = datetime.strptime(sowing_date_str, "%Y-%m-%d")
        days_since_sowing = (datetime.now() - sowing_date).days
        lifecycle = crop_data.get("lifecycle", {}).get("stages", [])
        for stage in lifecycle:
            start, end = map(int, stage["day_range"].split("-"))
            if start <= days_since_sowing <= end:
                return stage
        return {"stage": "Harvested/Post-Cycle", "advice": "Crop cycle complete."}
    except Exception as e:
        print(f"Error getting current advice: {e}")
        return {"stage": "Unknown", "advice": "Invalid sowing date format."}


def normalize_disease_label(raw_label):
    if not isinstance(raw_label, str):
        return "Unknown", "Unknown"
    parts = raw_label.split("___", 1)
    crop = parts[0].replace("_", " ").replace("(including sour)", "").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return crop, condition


def get_disease_guidance(condition):
    condition_lower = condition.lower()
    if "healthy" in condition_lower:
        return {
            "remedy": "No action needed. Continue regular monitoring and maintain good crop care.",
            "precaution": "Keep leaves dry, avoid overwatering, and continue routine inspections."
        }
    if "bacterial" in condition_lower:
        return {
            "remedy": "Remove infected leaves and treat with a copper-based bactericide if suitable for the crop.",
            "precaution": "Disinfect tools between plants and avoid overhead irrigation."
        }
    if "blight" in condition_lower:
        return {
            "remedy": "Prune affected tissue and apply an approved fungicide for blight control.",
            "precaution": "Improve air circulation and avoid wet foliage during evening hours."
        }
    if "rust" in condition_lower:
        return {
            "remedy": "Remove heavily infected leaves and use a rust-specific fungicide spray.",
            "precaution": "Space plants to reduce humidity and water at the soil level only."
        }
    if "mildew" in condition_lower:
        return {
            "remedy": "Apply a sulfate or neem-based spray and remove powdery mildew from leaf surfaces.",
            "precaution": "Avoid overcrowding plants and keep humidity low around the crop."
        }
    if "scorch" in condition_lower or "spot" in condition_lower:
        return {
            "remedy": "Remove affected leaves and consider applying a general-purpose fungicide or insecticide as needed.",
            "precaution": "Avoid leaf wetness and maintain a balanced nutrient program."
        }
    if "virus" in condition_lower or "mosaic" in condition_lower or "greening" in condition_lower:
        return {
            "remedy": "Isolate the plant and remove heavily infected tissue; virus infections often require crop replacement.",
            "precaution": "Control insect vectors and use certified disease-free seedlings."
        }
    if "spider mites" in condition_lower or "mite" in condition_lower:
        return {
            "remedy": "Use a horticultural oil or miticide labeled for spider mites and keep leaves sprayed with water.",
            "precaution": "Inspect plants regularly and maintain humidity levels to discourage mites."
        }
    return {
        "remedy": "Remove damaged leaves, keep plant health strong, and use an appropriate crop-safe treatment.",
        "precaution": "Keep a clean growing area and monitor regularly for changes in leaf appearance."
    }


def prepare_leaf_image(file_stream):
    image = Image.open(file_stream).convert("RGB")
    image = image.resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    if disease_model is None or not class_names:
        return jsonify({"error": "Disease model unavailable on server."}), 503

    uploaded = request.files.get("leaf_image")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "No image file uploaded. Please select a leaf photo."}), 400

    plant_id_api_key = os.environ.get("PLANT_ID_API_KEY")
    if plant_id_api_key and plant_id_api_key != "YOUR_API_KEY_HERE":
        try:
            import base64
            uploaded.stream.seek(0)
            image_data = uploaded.stream.read()
            encoded_image = base64.b64encode(image_data).decode("ascii")

            api_url = "https://api.plant.id/v2/health_assessment"
            payload = {
                "images": [encoded_image],
                "modifiers": ["crops_fast", "similar_images"],
                "disease_details": ["cause", "common_names", "classification", "description", "treatment"]
            }
            headers = {
                "Content-Type": "application/json",
                "Api-Key": plant_id_api_key
            }

            response = requests.post(api_url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                health_assessment = data.get("health_assessment", {})
                is_healthy = health_assessment.get("is_healthy", False)
                diseases = health_assessment.get("diseases", [])

                if is_healthy or not diseases:
                    return jsonify({
                        "prediction": "Healthy",
                        "crop": "Unknown Plant",
                        "condition": "Healthy",
                        "confidence": 1.0,
                        "remedy": "No action needed. Continue regular monitoring.",
                        "precaution": "Keep leaves dry and avoid overwatering.",
                        "is_healthy": True
                    })
                
                best_disease = diseases[0]
                disease_name = best_disease.get("name", "Unknown Disease")
                disease_prob = best_disease.get("probability", 0.0)
                
                treatment = "Maintain good crop care."
                disease_details = best_disease.get("disease_details", {})
                if disease_details and "treatment" in disease_details:
                    treatments = disease_details["treatment"]
                    treatment_list = []
                    for cat, items in treatments.items():
                        if items:
                            treatment_list.extend(items)
                    if treatment_list:
                        treatment = " ".join(treatment_list[:2])
                
                return jsonify({
                    "prediction": disease_name,
                    "crop": "Detected Plant",
                    "condition": disease_name,
                    "confidence": round(disease_prob, 3),
                    "remedy": treatment,
                    "precaution": "Monitor plant health and follow standard agricultural practices.",
                    "is_healthy": False
                })
            else:
                print(f"Plant.id API error: {response.status_code} - {response.text}")
                # Fallback to local model if API fails
        except Exception as e:
            print(f"Plant.id API exception: {e}")
            # Fallback to local model if exception occurs

    # --- Local Model Fallback ---
    if disease_model is None or not class_names:
        return jsonify({"error": "Disease model unavailable on server and Plant.id API key not configured."}), 503

    try:
        uploaded.stream.seek(0)
        image_data = prepare_leaf_image(uploaded.stream)
        predictions = disease_model.predict(image_data)
        prediction_vector = predictions[0]
        prediction_index = int(np.argmax(prediction_vector))
        confidence_score = float(np.max(prediction_vector))
        raw_label = label_lookup.get(prediction_index, "Unknown")
        crop, condition = normalize_disease_label(raw_label)
        guidance = get_disease_guidance(condition)

        return jsonify({
            "prediction": raw_label,
            "crop": crop,
            "condition": condition,
            "confidence": round(confidence_score, 3),
            "remedy": guidance["remedy"],
            "precaution": guidance["precaution"],
            "is_healthy": "healthy" in condition.lower()
        })
    except Exception as e:
        print(f"Disease prediction failed: {e}")
        return jsonify({"error": f"Disease prediction failed: {e}"}), 500


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/dashboard")
@app.route("/dashboard.html")
def dashboard_page():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/crop_recommendations")
@app.route("/crop_recommendations.html")
def crop_recommendations_page():
    return send_from_directory(BASE_DIR, "crop_recommendations.html")


# Fallback static routes for assets referenced by the recommendations page
@app.route('/crop_recommendations.js')
def serve_crop_recommendations_js():
    return send_from_directory(BASE_DIR, 'crop_recommendations.js')

@app.route('/crop_recommendations.css')
def serve_crop_recommendations_css():
    return send_from_directory(BASE_DIR, 'style.css')

@app.route('/price_predictor')
@app.route('/price_predictor.html')
def price_predictor_page():
    return send_from_directory(BASE_DIR, 'price_predictor.html')

@app.route('/price_predictor.js')
def serve_price_predictor_js():
    return send_from_directory(BASE_DIR, 'price_predictor.js')

@app.route("/ask_agronomist", methods=["POST"])
def ask_agronomist():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").lower()
    
    # Get live sensor data
    try:
        sensors = get_sensor_payload()
        moisture = sensors.get('moisture', 0)
        temp = sensors.get('temperature', 0)
        ph = sensors.get('pH', sensors.get('ph', 0))
        humidity = sensors.get('humidity', 0)
    except Exception as e:
        print(f"Error fetching sensor payload: {e}")
        moisture, temp, ph, humidity = 0, 0, 0, 0
    
    # Dynamic fallback responses based on live data
    import random
    
    responses = []
    
    # Check for greetings
    if any(w in user_message for w in ["hello", "hi", "hey", "greetings"]):
        responses.append(random.choice([
            "Hello! I am your AI Agronomist. I'm actively monitoring your live sensor data.",
            "Hi there! I have access to your field's live conditions. What would you like to know?",
            "Greetings! I'm ready to analyze your soil and weather metrics."
        ]))

    # Analyze specifically requested metrics, or summarize all if it's a general question
    is_general = any(w in user_message for w in ["status", "summary", "how is", "everything", "farm", "field"])
    
    if is_general or any(w in user_message for w in ["moisture", "water", "irrigation"]):
        if moisture < 45:
            responses.append(random.choice([
                f"The soil moisture is quite low at {moisture}%. You should start irrigation soon.",
                f"I'm detecting low moisture ({moisture}%). Watering is recommended to prevent stress.",
                f"Moisture levels have dropped to {moisture}%. It's a good time to turn on the sprinklers."
            ]))
        elif moisture > 65:
            responses.append(random.choice([
                f"Soil moisture is high ({moisture}%). Hold off on watering.",
                f"The ground is quite wet with {moisture}% moisture. No irrigation needed currently.",
                f"At {moisture}% moisture, the soil has plenty of water. Let it drain a bit."
            ]))
        else:
            responses.append(random.choice([
                f"Moisture is perfectly balanced at {moisture}%.",
                f"Your field has ideal water levels ({moisture}% moisture).",
                f"The soil moisture is looking great at {moisture}%."
            ]))
            
    if is_general or any(w in user_message for w in ["ph", "acid", "alkaline"]):
        if ph < 6.0:
            responses.append(random.choice([
                f"The soil pH is acidic ({ph}). Consider applying agricultural lime.",
                f"I'm seeing a low pH of {ph}. Lime application could help neutralize it.",
                f"Your pH is {ph}, which is a bit acidic for most standard crops."
            ]))
        elif ph > 7.5:
            responses.append(random.choice([
                f"The soil is alkaline (pH {ph}). Elemental sulfur can help lower it.",
                f"With a pH of {ph}, the soil is quite basic. Monitor for nutrient lock-out.",
                f"pH levels are high at {ph}. You might want to address this alkalinity."
            ]))
        else:
            responses.append(random.choice([
                f"Soil pH is excellent at {ph}.",
                f"Your pH is {ph}, right in the optimal range.",
                f"The pH level ({ph}) is perfect for nutrient absorption."
            ]))

    if is_general or any(w in user_message for w in ["temperature", "heat", "cold", "weather"]):
        if temp > 35:
            responses.append(random.choice([
                f"It's very hot out there ({temp}°C). Watch out for heat stress.",
                f"The temperature is high at {temp}°C. Ensure your crops have enough water.",
                f"At {temp}°C, heat stress is a major risk today."
            ]))
        elif temp < 15:
            responses.append(random.choice([
                f"It's a bit chilly at {temp}°C.",
                f"The temperature is cool ({temp}°C). Monitor for frost if it drops further.",
                f"At {temp}°C, growth rates might slow down slightly."
            ]))
        else:
            responses.append(random.choice([
                f"The temperature is comfortable at {temp}°C.",
                f"It's {temp}°C, which is generally very favorable for plant growth.",
                f"Current temperature is {temp}°C, providing good growing conditions."
            ]))
            
    if is_general or any(w in user_message for w in ["humidity"]):
        if humidity > 80:
            responses.append(random.choice([
                f"Humidity is high ({humidity}%). This increases fungal disease risks.",
                f"It's very humid ({humidity}%). Keep an eye out for mildew.",
                f"At {humidity}% humidity, air circulation is critical."
            ]))
        elif humidity < 40:
            responses.append(random.choice([
                f"The air is quite dry ({humidity}% humidity).",
                f"Low humidity ({humidity}%) might increase transpiration rates.",
                f"With only {humidity}% humidity, crops might need more water."
            ]))
        else:
            responses.append(random.choice([
                f"Humidity is well-balanced at {humidity}%.",
                f"The {humidity}% humidity level is safe for most crops.",
                f"Humidity is normal at {humidity}%."
            ]))

    # Specific non-sensor queries
    if "disease" in user_message:
        responses.append("If you suspect an infection, use our Leaf Disease Analyzer tool for a precise AI diagnosis.")
    elif "profit" in user_message or "recommend" in user_message:
        responses.append("Check the Crop Recommendations page for AI-driven ROI forecasts and suitability scores.")
    elif "drone" in user_message:
        responses.append("The autonomous drone patrol system is active and monitoring your field parameters.")
    elif "white dots" in user_message:
        responses.append("White dots often indicate powdery mildew or spider mites. I'd suggest applying neem oil.")

    if not responses:
        responses.append(random.choice([
            "Could you provide more details? I can analyze your specific soil moisture, pH, temperature, or humidity.",
            "I'm not quite sure what you mean. Would you like a summary of your live field conditions?",
            "Please ask me about your live sensor metrics (water, temperature, pH) or general farm status."
        ]))

    response_text = " ".join(responses)
            
    # Attempt to use real Gemini API if available (Hackathon stretch goal)
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            context = f"Current Field Conditions: Moisture: {moisture}%, Temperature: {temp}°C, pH: {ph}, Humidity: {humidity}%."
            prompt = f"Act as an expert agronomist. {context} The user asks: {user_message}. Give a concise, professional, 2-sentence response directly addressing their question using the provided data."
            ai_resp = model.generate_content(prompt)
            if ai_resp.text:
                response_text = ai_resp.text
    except Exception as e:
        print("Gemini API error:", e)

    return jsonify({"reply": response_text})

@app.route("/predict_crop_price", methods=["GET", "POST"])
def predict_crop_price():
    data = request.get_json(silent=True) or request.args.to_dict() or {}
    crop_name = data.get("crop_name")
    date_str = data.get("date")

    if not crop_name:
        return jsonify({"error": "Missing parameter: crop_name"}), 400

    # Normalize and resolve crop code
    normalized_crop = crop_name.lower().replace(" ", "").replace("_", "")
    crop_code = CROP_MAPPING.get(normalized_crop)
    
    if crop_code is None:
        # Fallback partial matching
        for key, val in CROP_MAPPING.items():
            if normalized_crop in key or key in normalized_crop:
                crop_code = val
                break

    if crop_code is None:
        return jsonify({"error": f"Crop '{crop_name}' is not supported in the mapping."}), 400

    target_month = datetime.now().month
    target_year = datetime.now().year
    if date_str:
        try:
            dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
            target_month = dt_obj.month
            target_year = dt_obj.year
        except ValueError:
            try:
                dt_obj = datetime.strptime(date_str, "%Y-%m")
                target_month = dt_obj.month
                target_year = dt_obj.year
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD or YYYY-MM."}), 400

    if market_model is not None and scaler is not None:
        try:
            # Predict the price using model and scaler (4 features: crop_code, month, state_code=1.0, year)
            raw_input = np.array([[float(crop_code), float(target_month), 1.0, float(target_year)]])
            scaled_input = scaler.transform(raw_input)
            predicted_price = float(market_model.predict(scaled_input)[0])
            return jsonify({
                "success": True,
                "crop_name": crop_name,
                "crop_code": crop_code,
                "target_date": date_str or datetime.now().strftime("%Y-%m-%d"),
                "target_month": target_month,
                "predicted_price": round(predicted_price, 2)
            })
        except Exception as e:
            return jsonify({"error": f"Model inference failure: {str(e)}"}), 500
    else:
        return jsonify({"error": "Machine Learning components are not initialized."}), 503


@app.route("/get_live_price", methods=["GET"])
def get_live_price():
    crop_name = request.args.get("crop_name", "").strip()
    if not crop_name:
        return jsonify({"error": "Missing parameter: crop_name"}), 400

    normalized_crop = crop_name.lower().replace(" ", "").replace("_", "")
    crop_code = CROP_MAPPING.get(normalized_crop)
    if crop_code is None:
        return jsonify({"error": f"Crop '{crop_name}' not found."}), 400

    api_key = os.getenv("DATAGOV_API_KEY")
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Try fetching from Open Government Data India API if key exists
    if api_key:
        try:
            url = f"https://api.data.gov.in/resource/9ef842c1-d395-4290-a7ec-d1390ede1a1d?api-key={api_key}&format=json&limit=1&filters[commodity]={crop_name.capitalize()}"
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                records = response.json().get("records", [])
                if records:
                    modal_price_quintal = float(records[0].get("modal_price", 0))
                    price_per_kg = modal_price_quintal / 100.0
                    if price_per_kg > 0:
                        return jsonify({
                            "success": True,
                            "crop_name": crop_name,
                            "live_price": round(price_per_kg, 2),
                            "source": "data.gov.in API",
                            "updated_at": today_str
                        })
        except Exception as e:
            print(f"API fetch error: {e}")

    # Baseline from ML prediction
    baseline_price = None
    if market_model is not None and scaler is not None:
        try:
            current_month = datetime.now().month
            current_year = datetime.now().year
            raw_input = np.array([[float(crop_code), float(current_month), 1.0, float(current_year)]])
            scaled_input = scaler.transform(raw_input)
            baseline_price = float(market_model.predict(scaled_input)[0])
        except Exception as e:
            print(f"Model prediction error: {e}")

    if baseline_price is None:
        price_ranges = {
            "wheat": (20.0, 35.0), "rice": (22.0, 38.0), "maize": (15.0, 25.0), "corn": (15.0, 25.0),
            "sorghum": (50.0, 70.0), "pearlmillet": (60.0, 90.0), "bajra": (60.0, 90.0),
            "fingermillet": (45.0, 65.0), "ragi": (45.0, 65.0), "barley": (35.0, 55.0),
            "cotton": (80.0, 150.0), "jute": (40.0, 120.0), "sugarcane": (2.0, 4.0),
            "coffee": (100.0, 200.0), "rubber": (80.0, 150.0), "sesame": (10.0, 20.0),
            "groundnut": (50.0, 80.0), "mustard": (60.0, 90.0), "coconut": (15.0, 35.0),
            "tobacco": (60.0, 90.0), "banana": (15.0, 30.0), "potato": (10.0, 25.0),
            "onion": (15.0, 35.0), "apple": (80.0, 150.0), "orange": (30.0, 60.0),
            "mango": (50.0, 100.0), "papaya": (20.0, 50.0), "grape": (30.0, 60.0),
            "pomegranate": (120.0, 180.0), "garlic": (40.0, 70.0), "ginger": (50.0, 85.0),
            "cardamom": (1200.0, 1800.0), "blackpepper": (400.0, 600.0), "nutmeg": (200.0, 300.0),
            "cloves": (600.0, 900.0), "cumin": (250.0, 350.0), "fennel": (90.0, 120.0),
            "coriander": (150.0, 200.0), "turmeric": (60.0, 90.0), "oats": (50.0, 80.0),
            "fenugreek": (30.0, 50.0), "tea": (400.0, 700.0), "cashew": (600.0, 900.0),
            "almond": (400.0, 600.0), "walnut": (300.0, 500.0), "soybean": (40.0, 70.0)
        }
        c_min, c_max = price_ranges.get(normalized_crop, (25.0, 60.0))
        baseline_price = (c_min + c_max) / 2.0

    # Apply daily fluctuation of up to +/- 3% based on date seed
    import random
    seed_val = int(hash(f"{normalized_crop}-{today_str}") % 1000000)
    random.seed(seed_val)
    noise_percent = random.uniform(-0.03, 0.03)
    live_price = baseline_price * (1.0 + noise_percent)

    return jsonify({
        "success": True,
        "crop_name": crop_name,
        "live_price": round(live_price, 2),
        "source": "Local Price Simulator (Daily Seeded)",
        "updated_at": today_str
    })


@app.route("/update_simulated_sensors", methods=["POST"])
def update_simulated_sensors():
    global MOCK_SENSORS
    data = request.get_json(silent=True) or {}
    if not data:
        MOCK_SENSORS = None
        return jsonify({"success": True, "message": "Simulated sensors cleared"})

    ph = data.get("ph")
    moisture = data.get("moisture")
    temperature = data.get("temperature")
    humidity = data.get("humidity")
    
    MOCK_SENSORS = {
        "ph": float(ph) if ph is not None else 6.5,
        "moisture": float(moisture) if moisture is not None else 50.0,
        "temperature": float(temperature) if temperature is not None else 25.0,
        "humidity": float(humidity) if humidity is not None else 65.0,
        "nitrogen": 45.0,
        "phosphorus": 30.0,
        "potassium": 120.0
    }
    payload = get_sensor_payload()
    socketio.emit('sensor_update', payload)
    return jsonify({"success": True, "payload": payload})


@app.route("/get_simulated_weather", methods=["GET"])
def get_simulated_weather():
    global MOCK_WEATHER
    if MOCK_WEATHER is None:
        MOCK_WEATHER = get_default_weather()
    return jsonify({"success": True, "weather": MOCK_WEATHER})


@lru_cache(maxsize=32)
def get_12_month_climate_forecast(lat, lon):
    """
    Fetches the past 12 months of historical weather data for the coordinates
    and aggregates it into a robust 12-month climate projection starting from the current month.
    """
    # Use a recent full year (e.g. 2023) as the baseline for the upcoming 12 months
    url = (f"https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}"
           f"&start_date=2023-01-01&end_date=2023-12-31"
           f"&daily=temperature_2m_mean,precipitation_sum"
           f"&timezone=auto")
    
    months_names = ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"]
                    
    # Fallback baseline in case API fails
    base_temps = [15.0, 19.0, 26.0, 32.0, 35.0, 33.0, 30.0, 29.0, 28.0, 26.0, 20.0, 16.0]
    base_rains = [10.0, 15.0, 12.0, 15.0, 25.0, 70.0, 190.0, 180.0, 120.0, 35.0, 10.0, 8.0]
    
    forecast_by_month = {m: {"temps": [], "rains": []} for m in range(12)}
    
    try:
        resp = requests.get(url, timeout=4.0)
        if resp.status_code == 200:
            daily = resp.json().get("daily", {})
            dates = daily.get("time", [])
            temps = daily.get("temperature_2m_mean", [])
            rains = daily.get("precipitation_sum", [])
            
            for i, date_str in enumerate(dates):
                m_idx = int(date_str[5:7]) - 1
                if i < len(temps) and temps[i] is not None:
                    forecast_by_month[m_idx]["temps"].append(float(temps[i]))
                if i < len(rains) and rains[i] is not None:
                    forecast_by_month[m_idx]["rains"].append(float(rains[i]))
    except Exception as e:
        print(f"Archive API error: {e}")

    current_month_idx = datetime.now().month - 1
    results = []
    
    for i in range(12):
        target_m_idx = (current_month_idx + i) % 12
        m_data = forecast_by_month.get(target_m_idx, {})
        
        avg_temp = base_temps[target_m_idx]
        if m_data.get("temps"):
            avg_temp = sum(m_data["temps"]) / len(m_data["temps"])
            
        total_rain = base_rains[target_m_idx]
        if m_data.get("rains"):
            total_rain = sum(m_data["rains"])
            
        results.append({
            "month": months_names[target_m_idx],
            "month_num": target_m_idx + 1,
            "temp": round(avg_temp, 1),
            "rain": round(total_rain, 1),
            "source": "live"
        })
        
    return results

@app.route("/get_daily_weather_forecast", methods=["GET"])
def get_daily_weather_forecast():
    """Fetches 7-day daily weather forecast using Open-Meteo."""
    lat = float(request.args.get("lat", 28.61))
    lon = float(request.args.get("lon", 77.20))
    days = int(request.args.get("days", 7))
    weather = get_weather_forecast(lat, lon, days)
    return jsonify({"success": True, "weather": weather})

@app.route("/get_live_weather_forecast", methods=["GET"])
def get_live_weather_forecast():
    """Fetches 12-month weather data using cached Open-Meteo forecast API handler."""
    lat = float(request.args.get("lat", 28.61))
    lon = float(request.args.get("lon", 77.20))
    weather = get_12_month_climate_forecast(lat, lon)
    return jsonify({"success": True, "weather": weather})


@app.route("/update_simulated_weather", methods=["POST"])
def update_simulated_weather():
    global MOCK_WEATHER
    data = request.get_json(silent=True) or {}
    weather_list = data.get("weather")
    if weather_list is None or not isinstance(weather_list, list):
        return jsonify({"error": "Invalid payload: weather must be a list"}), 400
    
    if len(weather_list) == 0:
        MOCK_WEATHER = None
        return jsonify({"success": True, "weather": get_default_weather()})
    
    MOCK_WEATHER = []
    for item in weather_list:
        MOCK_WEATHER.append({
            "month": str(item.get("month", "")),
            "month_num": int(item.get("month_num", 1)),
            "temp": float(item.get("temp", 25.0)),
            "rain": float(item.get("rain", 50.0))
        })
    return jsonify({"success": True, "weather": MOCK_WEATHER})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "firebase_connected": firebase_app is not None})


@app.route("/get_sensor_data", methods=["GET"])
def get_sensor_data():
    return jsonify(get_sensor_payload())


@app.route("/sensor_history", methods=["GET"])
def sensor_history():
    try:
        conn = sqlite3.connect(os.path.join(os.environ.get('TEMP', '.'), 'sensor_history.db'))
        c = conn.cursor()
        c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 50")
        rows = c.fetchall()
        conn.close()
        rows.reverse()
        data = [{"timestamp": r[0], "moisture": r[1], "temperature": r[2], "humidity": r[3], "ph": r[4]} for r in rows]
        return jsonify(data)
    except Exception:
        return jsonify([])


@app.route("/predict_irrigation", methods=["GET"])
def predict_irrigation():
    lat = request.args.get('lat', 28.61, type=float)
    lon = request.args.get('lon', 77.20, type=float)
    sensors = get_sensor_payload()
    moisture = sensors["moisture"]
    temperature = sensors["temperature"]
    humidity = sensors["humidity"]
    ph = sensors["ph"]

    irrigation_required = 1 if (moisture < 38 or humidity > 85 or temperature > 32) else 0
    confidence_score = 0.72

    if irrigation_model is not None:
        try:
            features = np.array([[moisture, temperature, humidity, ph]], dtype=float)
            prediction = float(irrigation_model.predict(features)[0])
            irrigation_required = 1 if prediction >= 0.5 else 0
            confidence_score = min(0.99, max(0.55, abs(prediction - 0.5) * 2 + 0.5))
        except Exception as e:
            print(f"Error processing irrigation model: {e}")

    # Weather Integration: Override irrigation if heavy rain is forecasted in next 3 days
    forecast = get_weather_forecast(lat=lat, lon=lon)
    rain_forecast = forecast.get("precipitation_sum", [])
    upcoming_rain = sum(rain_forecast[:3]) if rain_forecast else 0.0
    message = "Water needed" if irrigation_required else "Soil moisture is healthy"
    
    if irrigation_required and upcoming_rain > 10.0:
        irrigation_required = 0
        message = f"Heavy rain ({upcoming_rain}mm) expected. Skip irrigation."

    return jsonify({
        "irrigation_required": irrigation_required,
        "confidence_score": round(confidence_score, 2),
        "disease_risk": 1 if (humidity > 80 and temperature > 28) else 0,
        "fertilizer_status": 1 if moisture < 40 else 0,
        "message": message
    })

@lru_cache(maxsize=128)
def get_predicted_price(crop_code, current_month, current_year=2026):
    if market_model is not None and scaler is not None:
        raw_input = np.array([[float(crop_code), float(current_month), 1.0, float(current_year)]])
        scaled_input = scaler.transform(raw_input)
        return float(market_model.predict(scaled_input)[0])
    return None

@app.route("/recommend_crop", methods=["GET", "POST"])
def recommend_crop():
    data = request.get_json(silent=True) or request.args.to_dict() or {}
    sowing_date = data.get("sowing_date")
    try:
        lat = float(data.get("lat", 28.61))
        lon = float(data.get("lon", 77.20))
    except ValueError:
        return jsonify({"error": "Invalid coordinates provided"}), 400
    current_month = datetime.now().month
    current_year = datetime.now().year
    weather_override = data.get("weather_override")

    all_crops = get_db_value("crop_knowledge_base", {}) or {}
    
    if not all_crops:
        # Fallback to a static dataset if Firebase is unavailable
        all_crops = {
            "Wheat": {
                "crop_code": 0, "ideal_temp": 20, "base_price": 30, "base_yield": 40, "sowing_months": [10, 11, 12],
                "initial_cost": 5000, "labour_cost": 2500, "post_planting": "Water every 10–12 days during crown root initiation, jointing, flowering, and grain filling. Inspect regularly for rust diseases."
            },
            "Rice": {
                "crop_code": 1, "ideal_temp": 28, "base_price": 40, "base_yield": 50, "sowing_months": [5, 6, 7],
                "initial_cost": 8000, "labour_cost": 4500, "post_planting": "Maintain shallow flooding water levels (2–5cm) during active tillering. Hand weed or apply organic herbicides 25 days after transplanting.",
                "water_intensive": True, "min_rain": 800, "max_rain": 2000
            },
            "Corn": {
                "crop_code": 2, "ideal_temp": 25, "base_price": 35, "base_yield": 60, "sowing_months": [3, 4, 5, 6],
                "initial_cost": 4500, "labour_cost": 2000, "post_planting": "Sufficient side-dressing nitrogen application after 30 days. Maintain consistent moisture levels during silking and cob development."
            }
        }
        
    sensors = get_sensor_payload()
    hardware_status = "ONLINE" if sensors.get("is_online", False) else "OFFLINE"

    results = []

    for crop_name, bio in all_crops.items():
        ph = sensors.get("ph", 6.5)
        moisture = sensors.get("moisture", 50)
        temp = sensors.get("temperature", 25.0)
        humidity = sensors.get("humidity", 65.0)

        ph_match = 1.0 if 6.0 <= ph <= 7.5 else 0.4
        temp_score = max(0, 1 - (abs(temp - bio.get("ideal_temp", 25)) / 10))
        moisture_score = max(0, 1 - (abs(moisture - 50) / 50))

        disease_risk_penalty = 0.15 if (humidity > 80 and temp > 28) else 0.0
        water_stress_penalty = 0.2 if moisture < 35 else 0.0

        # ── Weather forecast integration (12-month) ──────────────────────
        lifecycle_days = bio.get("lifecycle_days", 90)
        lifecycle_months = max(1, min(12, lifecycle_days // 30))
        
        # Fetch the live 12-month climate forecast starting this month
        forecast_12 = weather_override if weather_override else get_12_month_climate_forecast(lat, lon)
        
        # Slice it for the precise crop life period
        crop_forecast = forecast_12[:lifecycle_months]

        climate = {}

        if crop_forecast:
            avg_forecast_temp  = sum(m["temp"] for m in crop_forecast) / len(crop_forecast)
            total_forecast_rain = sum(m["rain"] for m in crop_forecast)

            ideal_temp  = bio.get("ideal_temp", 25)
            
            # Logic Fix: Map missing min_rain/max_rain defaults based on water_intensive flag
            is_water_intensive = bio.get("water_intensive", False) or crop_name.lower() in ["rice", "sugarcane", "banana", "jute"]
            default_min = 800 if is_water_intensive else 50
            default_max = 2500 if is_water_intensive else 800
            
            min_rain    = bio.get("min_rain", default_min)
            max_rain    = bio.get("max_rain", default_max)

            # Temperature fit against precise lifecycle forecast
            if abs(avg_forecast_temp - ideal_temp) > 8:
                temp_score = max(0, temp_score - 0.25)
                bio.setdefault("measures", []).append("Temperature deviation expected; consider shading/mulching or greenhouse if possible.")
            elif abs(avg_forecast_temp - ideal_temp) > 4:
                temp_score = max(0, temp_score - 0.10)
                
            # Rainfall fit against precise lifecycle forecast
            if total_forecast_rain < min_rain:
                deficit_ratio = (min_rain - total_forecast_rain) / max(min_rain, 1)
                water_stress_penalty += min(0.3, deficit_ratio * 0.3)
                bio.setdefault("measures", []).append(f"Expected rainfall ({round(total_forecast_rain)}mm) is below minimum required ({min_rain}mm). Ensure irrigation is ready.")
            elif total_forecast_rain > max_rain:
                excess_ratio = (total_forecast_rain - max_rain) / max(max_rain, 1)
                water_stress_penalty += min(0.15, excess_ratio * 0.15)
                bio.setdefault("measures", []).append(f"Expected rainfall ({round(total_forecast_rain)}mm) is above maximum safe threshold ({max_rain}mm). Ensure deep drainage trenches are dug.")
                
            # Dynamic monthly checks for extreme events during the lifecycle
            for i, m in enumerate(crop_forecast):
                if m["rain"] > 250:
                    bio.setdefault("measures", []).append(f"High risk of flooding in month {i+1} ({m['month']}). Secure field borders.")
                if m["temp"] > 38:
                    bio.setdefault("measures", []).append(f"Extreme heat stress risk in month {i+1} ({m['month']}). Plan for frequent, light watering.")
        else:
            # Fallback: use climate trend
            climate = get_climate_trend(lat=lat, lon=lon, days=lifecycle_days)
            if climate["total_rain"] < 50.0 and bio.get("water_intensive", False):
                water_stress_penalty += 0.25
            if abs(climate["avg_temp"] - bio.get("ideal_temp", 25)) > 5:
                temp_score = max(0, temp_score - 0.2)

        suitability_score = (ph_match * 0.3) + (temp_score * 0.3) + (moisture_score * 0.4)
        suitability_score = max(0, suitability_score - disease_risk_penalty - water_stress_penalty)

        pred_p = get_predicted_price(bio.get("crop_code", 0), current_month, current_year)
        predicted_price = pred_p if pred_p is not None else float(bio.get("base_price", 40))

        # ADD INLINE MACHINE LEARNING INFERENCE
        normalized_crop = crop_name.lower().replace(" ", "").replace("_", "")
        crop_encoded = CROP_MAPPING.get(normalized_crop)
        if crop_encoded is not None and market_model is not None and scaler is not None:
            try:
                raw_input = np.array([[float(crop_encoded), float(current_month), 1.0, float(current_year)]])
                scaled_input = scaler.transform(raw_input)
                predicted_price = float(market_model.predict(scaled_input)[0])
            except Exception as e:
                print(f"Model prediction error: {e}")
                predicted_price = float(bio.get("base_price", 22.0))

        revenue_expected = (bio.get("base_yield", 50) * (suitability_score * 0.8)) * predicted_price
        advice_data = get_current_advice(bio, sowing_date) if sowing_date else None

        sowing_months = bio.get("sowing_months")
        if sowing_months and isinstance(sowing_months, list):
            window_status = "Open" if current_month in sowing_months else "Closed"
        else:
            # Stricter fallback: requires temp to be within 2 degrees for sowing
            window_status = "Open" if abs(climate.get("avg_temp", 25) - bio.get("ideal_temp", 25)) <= 2.0 else "Closed"

        results.append({
            "crop": crop_name,
            "suitability_score": round(suitability_score * 100, 1),
            "expected_profit": round(revenue_expected - (bio.get("initial_cost", 1200) + bio.get("labour_cost", 800)), 2),
            "fertilizer_info": bio.get("fertilizer", {"type": "N/A", "schedule": "N/A"}),
            "measures": bio.get("measures", []),
            "current_lifecycle_advice": advice_data,
            "sowing_window": window_status,
            "initial_cost": bio.get("initial_cost") if bio.get("initial_cost") is not None else int(1500 + abs(hash(crop_name)) % 3000),
            "labour_cost": bio.get("labour_cost") if bio.get("labour_cost") is not None else int(800 + abs(hash(crop_name)) % 2000),
            "post_planting": bio.get("post_planting") if bio.get("post_planting") is not None else f"Keep the soil adequately drained. Weed field manually 2–3 weeks after sowing. Schedule pest control for common local vulnerabilities of {crop_name}.",
            "lifecycle_timeline": bio.get("lifecycle", {}).get("stages", [
                {"stage": "Sowing", "day_range": "0-10", "advice": "Plant seeds at optimal depth."},
                {"stage": "Vegetative", "day_range": "11-40", "advice": "Ensure adequate watering and nitrogen."},
                {"stage": "Flowering", "day_range": "41-70", "advice": "Critical period for water. Monitor for pests."},
                {"stage": "Harvest", "day_range": "71-100", "advice": "Harvest when crop reaches maturity."}
            ])
        })

    # Sort: sowing window OPEN first, then by suitability score descending
    results.sort(key=lambda item: (
        0 if item["sowing_window"] == "Open" else 1,
        -item["suitability_score"]
    ))
    return jsonify({
        "security_status": hardware_status,
        "current_field_conditions": sensors,
        "recommendations": results,
    })


def log_sensor_history(sensors):
    try:
        conn = sqlite3.connect(os.path.join(os.environ.get('TEMP', '.'), 'sensor_history.db'))
        c = conn.cursor()
        c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?)", 
                  (sensors['timestamp'], sensors['moisture'], sensors['temperature'], sensors['humidity'], sensors['ph']))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to log sensor history: {e}")

last_log_time = 0

def background_sensor_task():
    global last_log_time
    while True:
        try:
            sensors = get_sensor_payload()
            socketio.emit('sensor_update', sensors)
            current = time.time()
            if current - last_log_time > 300: # Log every 5 minutes
                log_sensor_history(sensors)
                last_log_time = current
        except Exception as e:
            print(f"Watchdog error: {e}")
        time.sleep(1)

threading.Thread(target=background_sensor_task, daemon=True).start()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True, allow_unsafe_werkzeug=True)
