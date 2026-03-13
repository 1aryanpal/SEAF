import pickle as pkl
from pathlib import Path
import os
import random
import json
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import hashlib
import time
import sqlite3
from datetime import datetime
import requests
from dotenv import load_dotenv
load_dotenv()



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")


# ========================================
# PATHS & CONFIGURATION
# ========================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


USERS_FILE = DATA_DIR / "users.txt"
DASHBOARD_FILE = DATA_DIR / "dashboard_content.json"

SECRET_KEY = os.environ.get("SECRET_KEY", "your_strong_secret_key_here")

# Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = SECRET_KEY


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS url_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            status TEXT,
            safe_confidence REAL,
            phishing_confidence REAL,
            checked_at TEXT,
            ip TEXT,
            country TEXT,
            region TEXT,
            city TEXT,
            lat REAL,
            lon REAL
        )
    """)
    conn.commit()
    conn.close()


# Initialize DB when app starts
init_db()



def extract_url_features(url: str):
    """Convert a URL into numeric features consistent with the training phase."""
    import re
    import tldextract
    from urllib.parse import urlparse

    # Example numeric feature extraction (same as in training)
    parsed = urlparse(url)
    domain_info = tldextract.extract(url)
    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "has_https": int("https" in url.lower()),
        "domain_length": len(domain_info.domain),
        "subdomain_length": len(domain_info.subdomain),
        "path_length": len(parsed.path),
    }

    return list(features.values())  # Return numeric vector



# ========================================
# CONFIGURATION
# ========================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = DATA_DIR / "users.txt"
DASHBOARD_FILE = DATA_DIR / "dashboard_content.json"

SECRET_KEY = os.environ.get("SECRET_KEY", "your_strong_secret_key_here")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = SECRET_KEY

# ========================================
# DEFAULT DASHBOARD CONTENT
# ========================================
DASHBOARD_CONTENT = {
    "title": "Social Engineering Attack Framework",
    "description": (
        "This framework helps in detecting, mapping, and preventing various types of "
        "social engineering attacks such as phishing, vishing, impersonation, and more. "
        "Explore live data insights and verify suspicious URLs in real-time."
    ),
}

# ========================================
# MODEL LOADING
# ========================================
phishing_model = None
scada_model = None

try:
    # Load phishing model using pickle for compatibility
    import pickle
    with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as f:
        phishing_model = pickle.load(f)
    print("[INFO] Phishing model loaded successfully.")
except Exception as e:
    print("[WARN] Could not load phishing model:", e)

try:
    scada_model = joblib.load(os.path.join(MODELS_DIR, "scada_pipeline.pkl"))
    print("[INFO] SCADA model loaded successfully.")
except Exception as e:
    print("[WARN] Could not load SCADA model:", e)


# ========================================
# HELPERS
# ========================================

def is_authenticated():
    return "user" in session


def current_user():
    return session.get("user")


def require_admin():
    u = current_user()
    return u and u.get("role") == "admin"


def extract_url_features(url: str) -> List[int]:
    return [
        len(url),
        url.count("."),
        1 if "http" in url.lower() else 0,
        1 if url.count("/") > 3 else 0,
        1 if "@" in url else 0,
    ]


def dataframe_from_features(features: List[float], prefix="Feature"):
    names = [f"{prefix}_{i+1}" for i in range(len(features))]
    return pd.DataFrame([features], columns=names)


# ========================================
# USER MANAGEMENT
# ========================================
def load_users() -> Dict[str, Dict[str, str]]:
    users = {}
    if not USERS_FILE.exists():
        return users
    with USERS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) == 3:
                username, hashed_pw, role = parts
                users[username] = {"password": hashed_pw, "role": role}
    return users


def save_user(username: str, password: str, role: str = "user") -> bool:
    users = load_users()
    if username in users:
        return False
    hashed = generate_password_hash(password)
    with USERS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"{username}|{hashed}|{role}\n")
    return True


def authenticate_user(username: str, password: str) -> Optional[Dict[str, str]]:
    users = load_users()
    u = users.get(username)
    if u and check_password_hash(u["password"], password):
        return {"username": username, "role": u["role"]}
    return None


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    role = data.get("role", "user").strip()

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400

    if role not in ("admin", "user"):
        role = "user"

    if save_user(username, password, role):
        return jsonify({"success": True, "message": "User registered"})
    else:
        return jsonify({"success": False, "message": "User already exists"}), 409


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    user = authenticate_user(username, password)
    if not user:
        return jsonify({"success": False, "message": "Invalid username or password"}), 401

    session["user"] = user
    return jsonify({"success": True, "message": "Login successful", "user": user})


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"success": True, "message": "Logged out"})


@app.route("/whoami")
def whoami():
    return jsonify({"user": current_user()})


@app.route("/edit_dashboard", methods=["POST"])
def edit_dashboard():
    if not require_admin():
        return jsonify({"success": False, "message": "Admin privileges required"}), 403

    data = request.get_json(force=True)
    title = data.get("title", "").strip()
    description = data.get("description", "").strip()

    if not title:
        return jsonify({"success": False, "message": "Title required"}), 400

    payload = {"title": title, "description": description}
    with DASHBOARD_FILE.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    DASHBOARD_CONTENT.update(payload)
    return jsonify({"success": True, "message": "Dashboard updated", "dashboard": payload})

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/docs")
def docs():
    return render_template("docs.html")


# ========================================
# EXISTING ROUTES (unchanged)
# ========================================
@app.route("/")
def index():
    return render_template("index.html", user=session.get("user"), dashboard_content=DASHBOARD_CONTENT)





@app.route("/check_url", methods=["POST"])
def check_url():
    """
    Checks a given URL using the phishing detection model.
    Accepts JSON: { "url": "http://example.com" }
    """
    try:
        data = request.get_json(force=True)
        url = data.get("url", "").strip()

        if not url:
            return jsonify({"error": "URL is required"}), 400

        if phishing_model is None:
            return jsonify({"error": "Phishing model not loaded"}), 500

        # --- Import your feature extractor ---
        from feature import FeatureExtraction
        import numpy as np

        # --- Extract features and predict ---
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = phishing_model.predict(x)[0]
        y_proba = phishing_model.predict_proba(x)[0]

        # Use model.classes_ to correctly map probabilities
        classes = list(phishing_model.classes_)
        proba_map = {cls: prob for cls, prob in zip(classes, y_proba)}

        # Get probabilities for phishing and safe (regardless of order)
        phish_prob = proba_map.get(1, 0.0)      # if class 1 means phishing
        safe_prob = proba_map.get(0, 0.0)       # if class 0 means safe
        # If your dataset used -1 for phishing and 1 for safe, also handle that:
        if -1 in proba_map and 1 in proba_map:
            phish_prob = proba_map.get(-1, 0.0)
            safe_prob = proba_map.get(1, 0.0)

        # --- Convert prediction result ---
        if y_pred == 1:
            result = f"It is {safe_prob * 100:.2f}% safe to go."
            status = "Safe"
        else:
            result = f"Website is {phish_prob * 100:.2f}% unsafe to use."
            status = "Phishing"


        import socket, requests
        from urllib.parse import urlparse
        from datetime import datetime
        def get_ip_location(ip):
            try:
                res = requests.get(f"http://ip-api.com/json/{ip}").json()
                if res["status"] == "success":
                    return res
            except:
                pass
            return None

        try:
            domain = urlparse(url).netloc
            ip = socket.gethostbyname(domain)
            loc = get_ip_location(ip)
            country = loc.get("country") if loc else None
            region = loc.get("regionName") if loc else None
            city = loc.get("city") if loc else None
            lat = loc.get("lat") if loc else None
            lon = loc.get("lon") if loc else None


            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                      INSERT INTO url_logs (url, status, safe_confidence, phishing_confidence, checked_at, ip, country, region, city, lat, lon)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                      """, (url,
                            status,
                            round(safe_prob * 100, 2),
                            round(phish_prob * 100, 2),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            ip,
                            country,
                            region,
                            city,
                            lat,
                            lon
                            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print("[WARN] Could not fetch or save location:", e)




        return jsonify({
            "url": url,
            "prediction": int(y_pred),
            "status": status,
            "safe_confidence": round(safe_prob * 100, 2),
            "phishing_confidence": round(phish_prob * 100, 2),
            "message": result
        })

    except Exception as e:
        print("[ERROR] /check_url failed:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/map")
def show_map():
    import pandas as pd
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM url_logs WHERE lat IS NOT NULL AND lon IS NOT NULL", conn)
        conn.close()
        logs = df.to_dict(orient="records")
    except Exception as e:
        print("[ERROR] Could not load map data:", e)
        logs = []

    return render_template("map.html", logs=logs)







#AI CHAT ENDPOINT (OpenRouter)
@app.route("/chat_ai", methods=["POST"])
def chat_ai():
    if not is_authenticated():
        return jsonify({"error": "Please login first."}), 401
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    print("Loaded OpenRouter Key:", OPENROUTER_API_KEY)

    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "SEA Framework AI"
        }

        payload = {
            "model": "openai/gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 500,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cybersecurity assistant for a Social Engineering "
                        "Attack Framework dashboard. Explain clearly and concisely."
                    )
                },
                {"role": "user", "content": user_message}
            ]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            print("OpenRouter error:", response.text)
            return jsonify({"error": "AI service unavailable"}), 503

        result = response.json()
        ai_reply = result["choices"][0]["message"]["content"]

        return jsonify({"reply": ai_reply})

    except Exception as e:
        print("[ERROR] AI Chat failed:", e)
        return jsonify({"error": "AI request failed"}), 500

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response




# ========================================
# START SERVER
# ========================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
