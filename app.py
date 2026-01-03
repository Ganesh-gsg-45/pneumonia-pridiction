import streamlit as st
import numpy as np
from PIL import Image
import os
import warnings
import logging
from dotenv import load_dotenv

# Suppress TensorFlow and related warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # More aggressive suppression
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='keras')
warnings.filterwarnings('ignore', category=FutureWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning, module='absl')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='absl')

# Suppress specific TensorFlow/Keras logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# SambaNova API integration (optional)
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
OPENAI_AVAILABLE = False

if SAMBANOVA_API_KEY:
    try:
        import openai
        # Initialize OpenAI client with SambaNova endpoint
        client = openai.OpenAI(
            api_key=SAMBANOVA_API_KEY,
            base_url="https://api.sambanova.ai/v1"
        )
        OPENAI_AVAILABLE = True
    except ImportError:
        st.warning("OpenAI library not available. Chat features will be limited to built-in knowledge.")
else:
    st.info("SambaNova API key not found. Chat features will use built-in knowledge base.")

# Try to import TensorFlow
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow is not installed. Run: pip install tensorflow")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI + Smart Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .suggestion-item {
        background: white;
        color: #333;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pneumonia_model_best.h5")

# Load pneumonia detection model
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            return None
        
        with st.spinner("Loading pneumonia detection model..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image
def preprocess_image(image):
    img = image.convert("L")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 224, 224, 1)
    return img_array, img

# Predict
def predict(model, image):
    processed_img, display_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)[0][0]
    return prediction, display_img

# ═══════════════════════════════════════════════════════════════
# TOP 5 RECOMMENDATIONS SYSTEM (NEW!)
# ═══════════════════════════════════════════════════════════════

def get_top5_recommendations(result, confidence):
    """Generate Top 5 recommendations based on X-ray analysis"""
    
    if result == "PNEUMONIA":
        if confidence > 85:
            return {
                "severity": "HIGH CONFIDENCE",
                "color": "#dc3545",
                "icon": "🔴",
                "recommendations": [
                    "🏥 **Seek IMMEDIATE medical attention** - Visit ER or pulmonologist within 24 hours",
                    "📋 **Document symptoms** - Record fever, cough severity, breathing difficulty, chest pain",
                    "🩺 **Request comprehensive tests** - CBC, sputum culture, possibly CT scan for confirmation",
                    "💧 **Stay hydrated** - Drink 8-10 glasses of water daily to help thin mucus",
                    "🛏️ **Rest completely** - Avoid all strenuous activities, get adequate sleep"
                ]
            }
        elif confidence > 65:
            return {
                "severity": "MODERATE CONFIDENCE",
                "color": "#ff8c00",
                "icon": "🟠",
                "recommendations": [
                    "👨‍⚕️ **Schedule doctor visit** - See healthcare provider within 24-48 hours",
                    "🌡️ **Monitor temperature** - Check every 4-6 hours, keep symptom diary",
                    "💊 **Avoid self-medication** - Don't take antibiotics without prescription",
                    "🚭 **Avoid irritants** - Stay away from smoke, pollution, strong chemicals",
                    "😷 **Practice hygiene** - Wear mask around others, cover coughs, wash hands"
                ]
            }
        else:
            return {
                "severity": "LOW CONFIDENCE",
                "color": "#ffc107",
                "icon": "🟡",
                "recommendations": [
                    "📞 **Consult doctor** - Schedule appointment for professional evaluation",
                    "📊 **Get additional tests** - Consider second X-ray or other imaging",
                    "👀 **Watch for symptoms** - Monitor for fever, cough, breathing changes",
                    "💪 **Support immunity** - Eat healthy, stay hydrated, get rest",
                    "📝 **Keep records** - Document any symptom changes for doctor visit"
                ]
            }
    else:  # NORMAL
        if confidence > 85:
            return {
                "severity": "HIGH CONFIDENCE NORMAL",
                "color": "#28a745",
                "icon": "🟢",
                "recommendations": [
                    "✅ **Continue healthy habits** - Maintain current respiratory health practices",
                    "🏃‍♂️ **Regular exercise** - 30 minutes daily to strengthen lung capacity",
                    "🥗 **Balanced diet** - Include vitamin C, D, and zinc-rich foods",
                    "💉 **Stay vaccinated** - Annual flu shot, pneumonia vaccine (if eligible)",
                    "🩺 **Routine checkups** - Annual physical exam as recommended by doctor"
                ]
            }
        else:
            return {
                "severity": "LIKELY NORMAL",
                "color": "#90EE90",
                "icon": "🟢",
                "recommendations": [
                    "👨‍⚕️ **Follow up if symptoms** - See doctor if you develop cough or fever",
                    "🔍 **Consider second opinion** - Additional imaging may provide clarity",
                    "💪 **Maintain health** - Continue healthy lifestyle practices",
                    "🚭 **Avoid risk factors** - Don't smoke, limit pollution exposure",
                    "📅 **Schedule checkup** - Regular monitoring is always beneficial"
                ]
            }

# ═══════════════════════════════════════════════════════════════
# BUILT-IN KNOWLEDGE BASE (Fallback if API fails)
# ═══════════════════════════════════════════════════════════════

def get_builtin_response(question):
    """Fallback knowledge base for common questions"""
    q = question.lower()
    
    if "symptom" in q or "signs" in q:
        return """🩺 **Common Pneumonia Symptoms:**

**Primary Symptoms:**
- 🌡️ High fever (over 100.4°F / 38°C)
- 😮‍💨 Shortness of breath
- 💨 Rapid breathing
- 😷 Cough with mucus (yellow/green/bloody)
- 💔 Chest pain when breathing/coughing

**Other Symptoms:**
- Chills, sweating
- Fatigue, weakness
- Nausea, vomiting
- Confusion (in elderly)

**⚠️ Emergency Signs:**
Seek immediate help if: severe breathing difficulty, blue lips/face, persistent chest pain, high fever not responding to medication."""

    elif "treatment" in q or "cure" in q:
        return """💊 **Pneumonia Treatment:**

**Medical Treatment:**
1. **Antibiotics** (bacterial) - Prescribed by doctor only
2. **Antivirals** (viral) - Oseltamivir for flu
3. **Fever reducers** - Acetaminophen, Ibuprofen

**Home Care:**
- 💧 Drink 8-10 glasses water daily
- 🛏️ Get plenty of rest
- 🌡️ Monitor temperature
- 🍲 Eat nutritious meals
- 🚭 Avoid smoking

**⚠️ Never self-medicate with antibiotics!**"""

    elif "prevent" in q:
        return """🛡️ **Prevention Strategies:**

1. 💉 **Vaccines** - Pneumococcal, flu, COVID-19
2. 🧼 **Hand hygiene** - Wash 20+ seconds frequently
3. 😷 **Wear masks** - In crowded/high-risk areas
4. 🚭 **Don't smoke** - Damages lung defenses
5. 💪 **Boost immunity** - Healthy diet, exercise, sleep
6. 🏥 **Manage conditions** - Control asthma, diabetes"""

    elif "diet" in q or "food" in q:
        return """🥗 **Nutrition for Recovery:**

**Essential Foods:**
- 🍗 **Protein:** Chicken, fish, eggs, yogurt
- 🍊 **Vitamin C:** Citrus, berries, broccoli
- 🥕 **Vitamin A:** Carrots, sweet potatoes, greens
- 🦪 **Zinc:** Seafood, meat, beans
- 💧 **Fluids:** Water, herbal tea, soup

**Avoid:** Processed foods, excessive dairy, alcohol, sugar"""

    elif "recover" in q or "how long" in q:
        return """⏱️ **Recovery Timeline:**

- **Mild:** 1-3 weeks
- **Moderate/Severe:** 3-6 weeks
- **Full strength:** Several months

**Week 1:** Fever subsides, fatigue persists
**Week 2-3:** Cough improves, energy increases
**Week 4+:** Gradual return to normal"""

    elif "doctor" in q or "emergency" in q:
        return """🏥 **When to See Doctor:**

**See Doctor If:**
- Fever >102°F (39°C)
- Persistent shortness of breath
- Chest pain
- Coughing up blood
- Symptoms >3 weeks

**EMERGENCY (Call 911):**
🚨 Severe breathing difficulty
🚨 Blue lips/face
🚨 Confusion
🚨 Rapid heartbeat at rest
🚨 Chest pain with sweating"""

    else:
        return None  # Let LLM handle it

# ═══════════════════════════════════════════════════════════════
# LLM WITH FALLBACK
# ═══════════════════════════════════════════════════════════════

def ask_smart_assistant(user_message, pred_result=None, confidence=None):
    """Try LLM first, fallback to built-in knowledge"""
    
    # Try built-in knowledge first for common questions
    builtin = get_builtin_response(user_message)
    if builtin:
        return builtin + "\n\n---\n*Response from built-in knowledge base*"
    
    # Use LLM for complex/specific questions
    system_prompt = """You are a helpful medical information assistant specialized in pneumonia.
You provide evidence-based information but NEVER diagnose or replace doctors.
Always remind users to consult healthcare professionals for medical decisions.

Keep responses concise (under 300 words) and well-structured with emojis."""

    context = ""
    if pred_result and confidence:
        context = f"""
Current X-ray Analysis Context:
- AI Prediction: {pred_result}
- Confidence: {confidence:.1f}%
- Note: This is AI estimation, not medical diagnosis
"""

    messages = [
        {"role": "system", "content": system_prompt + context},
        {"role": "user", "content": user_message}
    ]

    try:
        with st.spinner("🤔 Thinking via SambaNova Cloud..."):
            response = client.chat.completions.create(
                model="Meta-Llama-3.3-70B-Instruct",
                messages=messages,
                temperature=0.7,
                max_tokens=400,
                stream=False
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"""⚠️ **LLM API Error:** {str(e)}

I'm having trouble connecting to the AI service. Please:
1. Check your internet connection
2. Verify API key and credits at https://cloud.sambanova.ai
3. Try asking common questions (symptoms, treatment, prevention) which I can answer without the API

Or try rephrasing your question!"""

# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.pred_result = None
    st.session_state.confidence = None
    st.session_state.recommendations = None

tab1, tab2 = st.tabs(["📸 X-Ray Analysis & Top 5 Recommendations", "🩺 Smart Medical Assistant"])

# ═══════════════════════════════════════════════════════════════
# TAB 1: X-RAY ANALYSIS + TOP 5 RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.title("🫁 Pneumonia Detection + Top 5 Recommendations")
    st.markdown("### Local AI CNN + Expert Recommendation System")
    
    st.warning("""
    **⚠️ IMPORTANT MEDICAL DISCLAIMER**  
    This is an EDUCATIONAL TOOL ONLY — NOT a medical device or diagnosis.  
    ALWAYS consult qualified healthcare professionals for medical decisions.  
    Do NOT use this for actual clinical diagnosis or treatment.
    """)

    with st.sidebar:
        st.header("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">95.2%</h3>
                    <p style="margin:0; font-size:14px;">AUC-ROC</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3 style="margin:0; color:#2ca02c;">87%</h3>
                    <p style="margin:0; font-size:14px;">Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.header("🔧 Technology Stack")
        st.info("""
        **Detection:** TensorFlow CNN (224×224 grayscale)
        **Chat AI:** SambaNova Cloud (Meta-Llama-3.3-70B)
        **Recommendations:** Expert rule-based system
        """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Chest X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose X-ray image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Upload a chest X-ray for pneumonia detection"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded X-Ray", width='stretch')

            if st.button("🔍 Analyze X-Ray & Get Recommendations", type="primary", width='stretch'):
                model = load_model()
                if model:
                    with st.spinner("🤖 Analyzing X-ray with AI..."):
                        try:
                            prob, _ = predict(model, image)
                            confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
                            result = "PNEUMONIA" if prob > 0.5 else "NORMAL"

                            # Generate recommendations
                            recommendations = get_top5_recommendations(result, confidence)

                            st.session_state.analysis_done = True
                            st.session_state.pred_result = result
                            st.session_state.confidence = confidence
                            st.session_state.recommendations = recommendations

                            st.success("✅ Analysis complete! See results →")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload a chest X-ray image to begin analysis")

    with col2:
        st.subheader("📊 Analysis Results")
        
        if st.session_state.analysis_done:
            result = st.session_state.pred_result
            conf = st.session_state.confidence
            rec = st.session_state.recommendations

            # Result Display
            st.markdown(f"""
                <div style="background:{rec['color']};color:white;padding:20px;border-radius:10px;margin:10px 0;">
                    <h2 style="margin:0;">{rec['icon']} {result}</h2>
                    <h3 style="margin:10px 0 0 0;">Confidence: {conf:.1f}%</h3>
                    <p style="margin:5px 0 0 0;font-size:14px;">{rec['severity']}</p>
                </div>
            """, unsafe_allow_html=True)

            # Progress bars
            st.progress(float(conf / 100), text=f"{result}: {conf:.1f}%")
            other_conf = 100 - conf
            other_label = "NORMAL" if result == "PNEUMONIA" else "PNEUMONIA"
            st.progress(float(other_conf / 100), text=f"{other_label}: {other_conf:.1f}%")

            st.markdown("---")

            # TOP 5 RECOMMENDATIONS (NEW!)
            st.markdown("""
                <div class="recommendation-box">
                    <h2 style="margin:0;">🎯 Top 5 Personalized Recommendations</h2>
                    <p style="margin:5px 0 0 0;">Based on your X-ray analysis results</p>
                </div>
            """, unsafe_allow_html=True)

            for i, suggestion in enumerate(rec['recommendations'], 1):
                st.markdown(f"""
                    <div class="suggestion-item">
                        <strong>{i}.</strong> {suggestion}
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.info("💡 **Next Step:** Visit the 'Smart Medical Assistant' tab to ask specific questions!")

        else:
            st.info("👈 Upload and analyze an X-ray to see results and recommendations")

# ═══════════════════════════════════════════════════════════════
# TAB 2: SMART MEDICAL ASSISTANT
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.title("🩺 Smart Medical Assistant")
    st.markdown("### Powered by SambaNova Cloud + Built-in Medical Knowledge")

    st.warning("**⚠️ Not medical advice** — Always consult real healthcare professionals")

    # Quick action buttons
    st.markdown("**💡 Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        ("🩺 Symptoms", "What are pneumonia symptoms?"),
        ("💊 Treatment", "What are treatment options?"),
        ("🛡️ Prevention", "How to prevent pneumonia?"),
        ("🥗 Diet", "What should I eat for recovery?")
    ]
    
    for i, (label, question) in enumerate(quick_questions):
        with [col1, col2, col3, col4][i]:
            if st.button(label, width='stretch'):
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({"role": "user", "content": question})
                response = ask_smart_assistant(
                    question,
                    st.session_state.pred_result if st.session_state.analysis_done else None,
                    st.session_state.confidence if st.session_state.analysis_done else None
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    st.markdown("---")

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """👋 **Hello! I'm your Smart Medical Assistant!**

I combine:
- 🤖 **SambaNova Cloud AI** (Meta-Llama-3.3-70B) for intelligent responses
- 📚 **Built-in medical knowledge** for instant answers to common questions

**I can help with:**
- Understanding your X-ray results (if you analyzed one)
- Pneumonia symptoms, treatment, prevention
- Diet and recovery advice
- When to see a doctor

**Try asking:**
- "Explain my X-ray result"
- "What are warning signs?"
- "How long does recovery take?"

Ask me anything! 💬"""}
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about pneumonia, symptoms, treatment, your X-ray..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        pred_result = st.session_state.pred_result if st.session_state.analysis_done else None
        confidence = st.session_state.confidence if st.session_state.analysis_done else None

        response = ask_smart_assistant(prompt, pred_result, confidence)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Footer
st.markdown("---")
st.caption("""
🫁 **Pneumonia Detection AI** | Local CNN + SambaNova Cloud + Expert Recommendations  
Educational Project • January 2026 • Not for clinical use
""")