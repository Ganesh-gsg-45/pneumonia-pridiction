import os, io, warnings, logging
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
print(f"[DEBUG] GROQ_API_KEY loaded: {'YES → ' + GROQ_API_KEY[:8] + '...' if GROQ_API_KEY else 'NOT FOUND ❌'}")
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

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

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

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', ai_connected=OPENAI_AVAILABLE)

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/analyze', methods=['POST'])
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
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
    print("🫁  PneumoVision — Flask")
    print(f"   Groq AI      : {'✅ Connected' if OPENAI_AVAILABLE else '❌ Set GROQ_API_KEY in .env'}")
    print(f"   Model        : {'✅ Found' if os.path.exists(MODEL_PATH) else '❌ Not found'}")
    app.run(debug=True, host='0.0.0.0', port=5000)
