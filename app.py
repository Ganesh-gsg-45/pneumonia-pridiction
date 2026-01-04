import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Environment variables - works with both local .env and Hugging Face Secrets
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass  # OK if dotenv not available

# Try to import requests for SambaNova API
try:
    import requests
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    st.error("Requests library not installed. Run: pip install requests")
    st.stop()

# Get API key from environment (works with HF Secrets)
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY", "")

# Check if running on Hugging Face
IS_HUGGINGFACE = os.getenv("SPACE_ID") is not None

if not SAMBANOVA_API_KEY:
    if IS_HUGGINGFACE:
        st.error("🔑 API Key not configured. Space owner: Please add SAMBANOVA_API_KEY to Space Secrets.")
    else:
        st.error("🔑 SAMBANOVA_API_KEY not found in environment!")
        st.info("""
        **Setup Instructions:**
        1. Create a .env file in the project root
        2. Add: SAMBANOVA_API_KEY=your_key_here
        3. Get your key from: https://cloud.sambanova.ai/apis
        """)
    st.warning("⚠️ App will run in LIMITED MODE (X-ray analysis only, no AI chat)")
    SAMBANOVA_API_KEY = None



# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow not installed. Run: pip install tensorflow")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
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

# Model path - works on both local and Hugging Face
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pneumonia_model_best.h5")

# Load pneumonia detection model
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            st.info("Please ensure the model file is in the 'models/' folder")
            return None
        
        with st.spinner("🔄 Loading pneumonia detection model..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
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

# Top 5 Recommendations System
def get_top5_recommendations(result, confidence):
    if result == "PNEUMONIA":
        if confidence > 85:
            return {
                "severity": "HIGH CONFIDENCE",
                "color": "#dc3545",
                "icon": "🔴",
                "recommendations": [
                    "🏥 **Seek IMMEDIATE medical attention** - Visit ER or pulmonologist within 24 hours",
                    "📋 **Document symptoms** - Record fever, cough severity, breathing difficulty, chest pain",
                    "🩺 **Request comprehensive tests** - CBC, sputum culture, possibly CT scan",
                    "💧 **Stay hydrated** - Drink 8-10 glasses of water daily",
                    "🛏️ **Rest completely** - Avoid strenuous activities, get adequate sleep"
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
                    "🚭 **Avoid irritants** - Stay away from smoke, pollution",
                    "😷 **Practice hygiene** - Wear mask, cover coughs, wash hands"
                ]
            }
        else:
            return {
                "severity": "LOW CONFIDENCE",
                "color": "#ffc107",
                "icon": "🟡",
                "recommendations": [
                    "📞 **Consult doctor** - Schedule appointment for evaluation",
                    "📊 **Get additional tests** - Consider second X-ray or imaging",
                    "👀 **Watch for symptoms** - Monitor for fever, cough, breathing changes",
                    "💪 **Support immunity** - Eat healthy, stay hydrated, rest",
                    "📝 **Keep records** - Document symptom changes"
                ]
            }
    else:  # NORMAL
        if confidence > 85:
            return {
                "severity": "HIGH CONFIDENCE NORMAL",
                "color": "#28a745",
                "icon": "🟢",
                "recommendations": [
                    "✅ **Continue healthy habits** - Maintain respiratory health",
                    "🏃‍♂️ **Regular exercise** - 30 minutes daily",
                    "🥗 **Balanced diet** - Vitamin C, D, zinc-rich foods",
                    "💉 **Stay vaccinated** - Flu shot, pneumonia vaccine",
                    "🩺 **Routine checkups** - Annual physical exam"
                ]
            }
        else:
            return {
                "severity": "LIKELY NORMAL",
                "color": "#90EE90",
                "icon": "🟢",
                "recommendations": [
                    "👨‍⚕️ **Follow up if symptoms** - See doctor if cough/fever develops",
                    "🔍 **Consider second opinion** - Additional imaging may help",
                    "💪 **Maintain health** - Continue healthy practices",
                    "🚭 **Avoid risk factors** - Don't smoke, limit pollution",
                    "📅 **Schedule checkup** - Regular monitoring is beneficial"
                ]
            }

# Built-in Knowledge Base
def get_builtin_response(question):
    q = question.lower()
    
    if "symptom" in q or "signs" in q:
        return """🩺 **Common Pneumonia Symptoms:**

**Primary Symptoms:**
- 🌡️ High fever (over 100.4°F / 38°C)
- 😮‍💨 Shortness of breath
- 💨 Rapid breathing
- 😷 Cough with mucus
- 💔 Chest pain when breathing

**⚠️ Emergency Signs:**
Seek immediate help if: severe breathing difficulty, blue lips/face, persistent chest pain."""

    elif "treatment" in q or "cure" in q:
        return """💊 **Pneumonia Treatment:**

**Medical Treatment:**
1. **Antibiotics** (bacterial) - Doctor prescription required
2. **Antivirals** (viral) - For influenza
3. **Fever reducers** - Acetaminophen, Ibuprofen

**Home Care:**
- 💧 Drink 8-10 glasses water daily
- 🛏️ Get plenty of rest
- 🌡️ Monitor temperature
- 🚭 Avoid smoking

⚠️ Never self-medicate with antibiotics!"""

    elif "prevent" in q:
        return """🛡️ **Prevention Strategies:**

1. 💉 **Vaccines** - Pneumococcal, flu, COVID-19
2. 🧼 **Hand hygiene** - Wash frequently (20+ seconds)
3. 😷 **Wear masks** - In crowded areas
4. 🚭 **Don't smoke** - Damages lungs
5. 💪 **Boost immunity** - Healthy diet, exercise, sleep"""

    elif "diet" in q or "food" in q:
        return """🥗 **Nutrition for Recovery:**

- 🍗 **Protein:** Chicken, fish, eggs, yogurt
- 🍊 **Vitamin C:** Citrus, berries, broccoli
- 🥕 **Vitamin A:** Carrots, sweet potatoes, greens
- 🦪 **Zinc:** Seafood, meat, beans
- 💧 **Fluids:** Water, herbal tea, soup"""

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

**EMERGENCY (Call 911):**
🚨 Severe breathing difficulty
🚨 Blue lips/face
🚨 Confusion
🚨 Chest pain with sweating

**See Doctor Soon:**
- Fever >102°F
- Persistent shortness of breath
- Coughing up blood"""

    else:
        return None

# Ask Smart Assistant
def ask_smart_assistant(user_message, pred_result=None, confidence=None):
    # Try built-in knowledge first
    builtin = get_builtin_response(user_message)
    if builtin:
        return builtin + "\n\n---\n*Response from built-in knowledge base*"
    
    # Use LLM if available
    if not LLM_AVAILABLE or not SAMBANOVA_API_KEY:
        return """⚠️ **AI Assistant Unavailable**

The AI chatbot requires a SambaNova API key. However, I can still answer common questions about:
- Symptoms
- Treatment
- Prevention
- Diet
- Recovery
- When to see a doctor

Try asking: "What are pneumonia symptoms?" or "How to prevent pneumonia?"

---
*To enable full AI chat, configure SAMBANOVA_API_KEY in Space Secrets*"""

    system_prompt = """You are a helpful medical information assistant specialized in pneumonia.
Provide evidence-based information but NEVER diagnose or replace doctors.
Always remind users to consult healthcare professionals.
Keep responses concise (under 300 words) and well-structured with emojis."""

    context = ""
    if pred_result and confidence:
        context = f"""
Current X-ray Analysis:
- Prediction: {pred_result}
- Confidence: {confidence:.1f}%
Note: AI estimation, not medical diagnosis
"""

    messages = [
        {"role": "system", "content": system_prompt + context},
        {"role": "user", "content": user_message}
    ]

    try:
        with st.spinner("🤔 Thinking via SambaNova Cloud..."):
            response = requests.post(
                "https://api.sambanova.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Meta-Llama-3.3-70B-Instruct",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 400,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"""⚠️ **LLM API Error:** {str(e)}

Falling back to built-in responses. Try asking:
- "What are pneumonia symptoms?"
- "How to prevent pneumonia?"
- "When should I see a doctor?"

---
*Check API key configuration and internet connection*"""

# Initialize session state
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.pred_result = None
    st.session_state.confidence = None
    st.session_state.recommendations = None

# Main App
tab1, tab2 = st.tabs(["📸 X-Ray Analysis & Top 5 Recommendations", "🩺 Smart Medical Assistant"])

# TAB 1: X-RAY ANALYSIS
with tab1:
    st.title("🫁 Pneumonia Detection + Top 5 Recommendations")
    st.markdown("### AI-Powered Medical Image Analysis")
    
    st.warning("""
    **⚠️ IMPORTANT MEDICAL DISCLAIMER**  
    This is an EDUCATIONAL TOOL ONLY — NOT a medical device.  
    ALWAYS consult qualified healthcare professionals for medical decisions.
    """)

    with st.sidebar:
        st.header("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card"><h3 style="margin:0; color:#1f77b4;">95.2%</h3><p style="margin:0; font-size:14px;">AUC-ROC</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3 style="margin:0; color:#2ca02c;">87%</h3><p style="margin:0; font-size:14px;">Accuracy</p></div>', unsafe_allow_html=True)
        
        st.header("🔧 Technology")
        st.info("**Detection:** TensorFlow CNN\n**Chat:** SambaNova Cloud\n**Recommendations:** Expert system")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Chest X-Ray")
        uploaded_file = st.file_uploader("Choose X-ray (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded X-Ray", use_container_width=True)

            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                model = load_model()
                if model:
                    with st.spinner("🤖 Analyzing..."):
                        try:
                            prob, _ = predict(model, image)
                            confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
                            result = "PNEUMONIA" if prob > 0.5 else "NORMAL"
                            recommendations = get_top5_recommendations(result, confidence)

                            st.session_state.analysis_done = True
                            st.session_state.pred_result = result
                            st.session_state.confidence = confidence
                            st.session_state.recommendations = recommendations

                            st.success("✅ Analysis complete!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload X-ray to begin")

    with col2:
        st.subheader("📊 Analysis Results")
        
        if st.session_state.analysis_done:
            result = st.session_state.pred_result
            conf = st.session_state.confidence
            rec = st.session_state.recommendations

            st.markdown(f'<div style="background:{rec["color"]};color:white;padding:20px;border-radius:10px;margin:10px 0;"><h2 style="margin:0;">{rec["icon"]} {result}</h2><h3 style="margin:10px 0 0 0;">Confidence: {conf:.1f}%</h3><p style="margin:5px 0 0 0;font-size:14px;">{rec["severity"]}</p></div>', unsafe_allow_html=True)

            st.progress(float(conf / 100), text=f"{result}: {conf:.1f}%")

            st.markdown("---")
            st.markdown('<div class="recommendation-box"><h2 style="margin:0;">🎯 Top 5 Personalized Recommendations</h2><p style="margin:5px 0 0 0;">Based on your analysis</p></div>', unsafe_allow_html=True)

            for i, suggestion in enumerate(rec['recommendations'], 1):
                st.markdown(f'<div class="suggestion-item"><strong>{i}.</strong> {suggestion}</div>', unsafe_allow_html=True)

            st.info("💡 Visit 'Smart Medical Assistant' tab for questions!")
        else:
            st.info("👈 Upload and analyze X-ray to see results")

# TAB 2: MEDICAL ASSISTANT
with tab2:
    st.title("🩺 Smart Medical Assistant")
    st.markdown("### AI-Powered Medical Information Guide")

    st.warning("⚠️ Not medical advice — Consult healthcare professionals")

    # Quick actions
    st.markdown("**💡 Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_q = [
        ("🩺 Symptoms", "What are pneumonia symptoms?"),
        ("💊 Treatment", "What are treatment options?"),
        ("🛡️ Prevention", "How to prevent pneumonia?"),
        ("🥗 Diet", "What should I eat?")
    ]
    
    for i, (label, question) in enumerate(quick_q):
        with [col1, col2, col3, col4][i]:
            if st.button(label, use_container_width=True):
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
        ai_status = "🤖 SambaNova Cloud AI" if LLM_AVAILABLE and SAMBANOVA_API_KEY else "📚 Built-in Knowledge Only"
        st.session_state.messages = [
            {"role": "assistant", "content": f"""👋 **Hello! I'm your Smart Medical Assistant!**

**Status:** {ai_status}

I can help with:
- Understanding X-ray results
- Pneumonia symptoms, treatment, prevention
- Diet and recovery advice
- When to see a doctor

Ask me anything! 💬"""}
        ]

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about pneumonia or your X-ray..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = ask_smart_assistant(
            prompt,
            st.session_state.pred_result if st.session_state.analysis_done else None,
            st.session_state.confidence if st.session_state.analysis_done else None
        )

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Footer
st.markdown("---")
if IS_HUGGINGFACE:
    st.caption("🫁 Pneumonia Detection AI | Running on Hugging Face Spaces | Educational Project • January 2026")
else:
    st.caption("🫁 Pneumonia Detection AI | Local CNN + SambaNova Cloud | Educational Project • January 2026")