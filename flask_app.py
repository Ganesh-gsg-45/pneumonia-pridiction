import os, io, warnings, logging, json, sys

# Fix Windows terminal Unicode encoding issue (emojis crash cp1252 terminals)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from functools import wraps
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# ── Suppress warnings ─────────────────────────────────────────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ── Load env ──────────────────────────────────────────────────────────────────
_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_ENV, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"[DEBUG] GROQ_API_KEY loaded: {'YES' if GROQ_API_KEY else 'NOT FOUND ❌'}")
OPENAI_AVAILABLE = False
client = None

if GROQ_API_KEY:
    try:
        import openai
        client = openai.OpenAI(api_key=GROQ_API_KEY,
                               base_url="https://api.groq.com/openai/v1")
        OPENAI_AVAILABLE = True
    except ImportError:
        pass



import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime
from authlib.integrations.flask_client import OAuth

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
# IMPORTANT: os.urandom(24) changes on every restart, invalidating sessions
# and causing CSRF mismatches in OAuth. Use a stable fallback instead.
_raw_secret = os.environ.get('SECRET_KEY')
if not _raw_secret:
    import hashlib
    _raw_secret = hashlib.sha256(b'pneumovision-stable-dev-key-2024').hexdigest()
    print("[WARNING] SECRET_KEY not set in .env — using stable fallback. Set SECRET_KEY in .env for production.")
app.secret_key = _raw_secret
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Correctly interpret requests coming through Hugging Face's reverse proxy
# (scheme, host) — important for reliable session cookie behavior behind
# a proxy that terminates HTTPS in front of the container.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
# Secure cookies required for HTTPS (Hugging Face Spaces uses HTTPS)
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('HF_SPACE_ID') is not None

# ── Google OAuth setup ────────────────────────────────────────────────────────
oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# ── Database Setup (PostgreSQL with SQLite fallback) ──────────────────────────
_raw_db_url = os.getenv("DATABASE_URL", "").strip()

# Normalise legacy postgres:// scheme used by Heroku / some cloud providers
if _raw_db_url.startswith("postgres://"):
    _raw_db_url = _raw_db_url.replace("postgres://", "postgresql://", 1)

# If the env var is missing, empty, or not a valid scheme → use SQLite
if not _raw_db_url or not (_raw_db_url.startswith("postgresql://") or
                            _raw_db_url.startswith("sqlite://")):
    print("[INFO] DATABASE_URL not set or invalid — using SQLite (sqlite:///pneumovision.db).")
    db_url = "sqlite:///pneumovision.db"
else:
    db_url = _raw_db_url

# If a PostgreSQL URL was given, verify the server is actually reachable
if db_url.startswith("postgresql://"):
    try:
        import psycopg2
        from urllib.parse import urlparse
        parsed = urlparse(db_url)
        conn = psycopg2.connect(
            dbname=parsed.path[1:] if parsed.path else 'postgres',
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname or 'localhost',
            port=parsed.port or 5432,
            connect_timeout=2
        )
        conn.close()
        print("[DEBUG] PostgreSQL database connection successful!")
    except Exception as pg_err:
        print(f"[WARNING] PostgreSQL connection failed ({pg_err}).")
        print("[INFO] Falling back to SQLite ('sqlite:///pneumovision.db').")
        db_url = "sqlite:///pneumovision.db"

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

if db_url.startswith("postgresql://"):
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 280,
    }

db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=True)   # nullable: Google-only users have no password
    google_id = db.Column(db.String(255), unique=True, nullable=True)  # Google OAuth
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

with app.app_context():
    db.create_all()
    # ── Schema migration: add google_id if it was added after the DB was created ──
    try:
        from sqlalchemy import text, inspect as sa_inspect
        inspector = sa_inspect(db.engine)
        existing_cols = [col['name'] for col in inspector.get_columns('users')]
        if 'google_id' not in existing_cols:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN google_id VARCHAR(255) UNIQUE"))
                conn.commit()
            print("[DEBUG] Migration applied: added 'google_id' column to users table.")
    except Exception as _mig_err:
        print(f"[WARNING] Migration check failed (non-fatal): {_mig_err}")
    print("[DEBUG] Database tables created/verified successfully.")

# ── login_required decorator ──────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            if request.path.startswith('/api/') or request.path in ('/analyze', '/predict', '/chat'):
                return jsonify({'error': 'Please log in to continue.'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return wrapper

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "pneumonia_model_best.h5")
_model = None

# ── Backend functions (unchanged logic from app.py) ───────────────────────────
def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def preprocess_image(image):
    img = image.convert("L").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 224, 224, 1)

def predict_image(image):
    prob = float(get_model().predict(preprocess_image(image), verbose=0)[0][0])
    result = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    confidence = round((prob if prob > 0.5 else 1 - prob) * 100, 1)
    return result, confidence

def get_top5_recommendations(result, confidence):
    if result == "PNEUMONIA":
        if confidence > 85:
            return {"severity": "HIGH CONFIDENCE", "color": "#ef4444", "icon": "🔴",
                    "recommendations": [
                        "🏥 **Seek IMMEDIATE medical attention** – Visit ER or pulmonologist within 24 hours",
                        "📋 **Document symptoms** – Record fever, cough severity, breathing difficulty, chest pain",
                        "🩺 **Request comprehensive tests** – CBC, sputum culture, possibly CT scan",
                        "💧 **Stay hydrated** – Drink 8–10 glasses of water daily to help thin mucus",
                        "🛏️ **Rest completely** – Avoid all strenuous activities, get adequate sleep"]}
        elif confidence > 65:
            return {"severity": "MODERATE CONFIDENCE", "color": "#f59e0b", "icon": "🟠",
                    "recommendations": [
                        "👨‍⚕️ **Schedule doctor visit** – See healthcare provider within 24–48 hours",
                        "🌡️ **Monitor temperature** – Check every 4–6 hours, keep symptom diary",
                        "💊 **Avoid self-medication** – Don't take antibiotics without prescription",
                        "🚭 **Avoid irritants** – Stay away from smoke, pollution, strong chemicals",
                        "😷 **Practice hygiene** – Wear mask around others, cover coughs, wash hands"]}
        else:
            return {"severity": "LOW CONFIDENCE", "color": "#eab308", "icon": "🟡",
                    "recommendations": [
                        "📞 **Consult doctor** – Schedule appointment for professional evaluation",
                        "📊 **Get additional tests** – Consider second X-ray or other imaging",
                        "👀 **Watch for symptoms** – Monitor for fever, cough, breathing changes",
                        "💪 **Support immunity** – Eat healthy, stay hydrated, get rest",
                        "📝 **Keep records** – Document any symptom changes for doctor visit"]}
    else:
        if confidence > 85:
            return {"severity": "HIGH CONFIDENCE NORMAL", "color": "#10b981", "icon": "🟢",
                    "recommendations": [
                        "✅ **Continue healthy habits** – Maintain current respiratory health practices",
                        "🏃 **Regular exercise** – 30 minutes daily to strengthen lung capacity",
                        "🥗 **Balanced diet** – Include vitamin C, D, and zinc-rich foods",
                        "💉 **Stay vaccinated** – Annual flu shot, pneumonia vaccine (if eligible)",
                        "🩺 **Routine checkups** – Annual physical exam as recommended by doctor"]}
        else:
            return {"severity": "LIKELY NORMAL", "color": "#34d399", "icon": "🟢",
                    "recommendations": [
                        "👨‍⚕️ **Follow up if symptoms** – See doctor if you develop cough or fever",
                        "🔍 **Consider second opinion** – Additional imaging may provide clarity",
                        "💪 **Maintain health** – Continue healthy lifestyle practices",
                        "🚭 **Avoid risk factors** – Don't smoke, limit pollution exposure",
                        "📅 **Schedule checkup** – Regular monitoring is always beneficial"]}

def get_builtin_response(q):
    q = q.lower()
    if "symptom" in q or "sign" in q:
        return "🩺 **Pneumonia Symptoms:**\n- 🌡️ High fever (>100.4°F)\n- 😮 Shortness of breath\n- 😷 Cough with mucus\n- 💔 Chest pain\n- Chills, fatigue, nausea\n\n⚠️ **Emergency:** Severe breathing difficulty, blue lips → call 911."
    if "treatment" in q or "cure" in q:
        return "💊 **Treatment:**\n1. Antibiotics (bacterial, doctor only)\n2. Antivirals (viral)\n3. Fever reducers\n\n🏠 **Home care:** Hydrate, rest, monitor temperature.\n\n⚠️ Never self-medicate with antibiotics!"
    if "prevent" in q:
        return "🛡️ **Prevention:**\n1. 💉 Vaccines (pneumococcal, flu)\n2. 🧼 Wash hands 20+ seconds\n3. 😷 Wear masks in crowds\n4. 🚭 Don't smoke\n5. 💪 Healthy diet & exercise"
    if "diet" in q or "food" in q:
        return "🥗 **Recovery Foods:**\n- 🍗 Protein: chicken, fish, eggs\n- 🍊 Vitamin C: citrus, berries\n- 💧 Fluids: water, herbal tea, soup\n\nAvoid: alcohol, sugar, processed foods."
    if "recover" in q or "how long" in q:
        return "⏱️ **Recovery:**\n- Mild: 1–3 weeks\n- Severe: 3–6 weeks\n- Full strength: several months"
    if "doctor" in q or "emergency" in q:
        return "🚨 **Call 911 if:**\n- Severe breathing difficulty\n- Blue lips or face\n- Confusion\n- Chest pain with sweating"
    return None

def ask_smart_assistant(message, pred_result=None, confidence=None):
    builtin = get_builtin_response(message)
    if builtin:
        return builtin + "\n\n---\n*Built-in knowledge base*"

    if not OPENAI_AVAILABLE or client is None:
        return "⚠️ AI chat not connected. Add `GROQ_API_KEY` to `.env` to enable.\n\nTry: symptoms, treatment, prevention, diet, or recovery."

    context = f"\nX-ray result: {pred_result} ({confidence:.1f}% confidence)\n" if pred_result and confidence else ""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a medical info assistant for pneumonia. Be concise, use emojis, never diagnose. Always advise consulting doctors." + context},
                {"role": "user", "content": message}
            ],
            temperature=0.7, max_tokens=400, stream=False)
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ API Error: {e}\n\nTry asking about symptoms, treatment, or prevention."


# ── Page Routes (login / signup pages) ──────────────────────────────────────
@app.route('/login')
def login_page():
    if session.get('user_id'):
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/signup')
def signup_page():
    if session.get('user_id'):
        return redirect(url_for('index'))
    return render_template('signup.html')


@app.route('/login/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/login/google/callback')
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
    except Exception as oauth_err:
        print(f"[WARNING] Google OAuth callback error: {oauth_err}")
        from flask import flash
        try:
            flash('Google sign-in failed or session expired. Please try again.', 'error')
        except Exception:
            pass
        return redirect(url_for('login_page'))
    userinfo = token.get('userinfo')
    if not userinfo:
        return redirect(url_for('login_page'))

    google_id = userinfo['sub']
    email = userinfo['email'].lower()
    name = userinfo.get('name', '') or email.split('@')[0]

    user = User.query.filter_by(google_id=google_id).first()

    if not user:
        # Check if this email already has a password-based account; link Google to it
        user = User.query.filter_by(email=email).first()
        if user:
            user.google_id = google_id
            db.session.commit()
        else:
            # New account via Google — generate a unique username
            base_username = name.replace(' ', '').lower()[:70] or email.split('@')[0]
            username = base_username
            suffix = 1
            while User.query.filter_by(username=username).first():
                suffix += 1
                username = f"{base_username}{suffix}"

            user = User(username=username, email=email, google_id=google_id, password_hash=None)
            db.session.add(user)
            db.session.commit()

    session['user_id'] = user.id
    session['username'] = user.username
    return redirect(url_for('index'))


# ── Auth API Routes ───────────────────────────────────────────────────────────
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({'error': 'Username, email, and password are required'}), 400

    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({'error': 'Username or Email already registered'}), 409

    hashed_pw = generate_password_hash(password)
    new_user = User(username=username, email=email, password_hash=hashed_pw)

    try:
        db.session.add(new_user)
        db.session.commit()
        session['user_id'] = new_user.id
        session['username'] = new_user.username
        return jsonify({'message': 'Registration successful', 'user': new_user.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    identifier = data.get('identifier', '').strip().lower()
    password = data.get('password', '')

    if not identifier or not password:
        return jsonify({'error': 'Username/Email and password are required'}), 400

    user = User.query.filter((User.email == identifier) | (User.username == identifier)).first()

    if not user or not user.password_hash or not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    session['user_id'] = user.id
    session['username'] = user.username
    return jsonify({'message': 'Login successful', 'user': user.to_dict()})

@app.route('/api/logout', methods=['POST', 'GET'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'})

@app.route('/api/me', methods=['GET'])
def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'logged_in': False})
    user = db.session.get(User, user_id)
    if not user:
        session.clear()
        return jsonify({'logged_in': False})
    return jsonify({'logged_in': True, 'user': user.to_dict()})

# ── Application Routes (protected) ───────────────────────────────────────────
@app.route('/')
@login_required
def index():
    user_id = session.get('user_id')
    user = db.session.get(User, user_id) if user_id else None
    return render_template('index.html', ai_connected=OPENAI_AVAILABLE, current_user=user.to_dict() if user else None)

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Only PNG/JPG/JPEG allowed'}), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
        result, confidence = predict_image(image)
        rec = get_top5_recommendations(result, confidence)
        return jsonify({'result': result, 'confidence': confidence,
                        'severity': rec['severity'], 'color': rec['color'],
                        'icon': rec['icon'], 'recommendations': rec['recommendations']})
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Image processing failed, please try again'}), 500

@app.route('/predict', methods=['POST'])
@login_required
def predict_alias():
    return analyze()

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    if not data or not data.get('message', '').strip():
        return jsonify({'error': 'Empty message'}), 400
    try:
        response = ask_smart_assistant(
            data['message'].strip(),
            data.get('pred_result'),
            data.get('confidence'))
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    print("PneumoVision -- Flask App")
    print(f"   Groq AI      : {'[OK] Connected' if OPENAI_AVAILABLE else '[!] Set GROQ_API_KEY in .env'}")
    print(f"   Model        : {'[OK] Found' if os.path.exists(MODEL_PATH) else '[!] Not found'}")
    print(f"   SECRET_KEY   : {'[OK] Set from environment' if os.environ.get('SECRET_KEY') else '[!] NOT SET -- using a random key, sessions will break on restart!'}")
    print(f"   Listening on : 0.0.0.0:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)