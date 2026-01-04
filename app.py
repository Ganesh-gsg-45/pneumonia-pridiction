import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress absl warnings

# Suppress TensorFlow deprecation warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import streamlit as st
import numpy as np
from PIL import Image
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

# Preprocess image - Grayscale for current model
def preprocess_image(image):
    img = image.convert("L")  # Grayscale
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 224, 224, 1)  # 1 channel for grayscale
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

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINED MEDICAL KNOWLEDGE BASE FOR LLM CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

MEDICAL_TRAINING_CONTEXT = """
=== PNEUMONIA MEDICAL KNOWLEDGE BASE ===

[SECTION 1: PNEUMONIA OVERVIEW]
Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.

Types of Pneumonia:
1. Bacterial Pneumonia: Most common, caused by Streptococcus pneumoniae
2. Viral Pneumonia: Caused by influenza, RSV, COVID-19
3. Mycoplasma Pneumonia: "Walking pneumonia" - milder form
4. Aspiration Pneumonia: From inhaling food, liquid, or vomit
5. Hospital-Acquired Pneumonia (HAP): Develops during hospital stay
6. Community-Acquired Pneumonia (CAP): Contracted outside healthcare settings

[SECTION 2: SYMPTOMS BY SEVERITY]

Mild Symptoms:
- Low-grade fever (99-100.4°F / 37.2-38°C)
- Mild cough with clear mucus
- Slight fatigue
- Minor chest discomfort

Moderate Symptoms:
- Fever (100.4-102°F / 38-39°C)
- Productive cough with yellow/green mucus
- Moderate shortness of breath during activity
- Fatigue and weakness
- Loss of appetite
- Night sweats

Severe Symptoms (EMERGENCY):
- High fever (>102°F / 39°C) or hypothermia
- Severe breathing difficulty at rest
- Rapid breathing (>30 breaths/minute)
- Confusion or altered mental status
- Cyanosis (blue lips, fingernails)
- Chest pain with each breath
- Blood in sputum
- Low blood pressure

[SECTION 3: DIAGNOSIS METHODS]
1. Chest X-ray: Primary diagnostic tool, shows lung infiltrates
2. CT Scan: More detailed imaging for complex cases
3. Blood tests: CBC, inflammatory markers (CRP, procalcitonin)
4. Sputum culture: Identifies causative organism
5. Pulse oximetry: Measures blood oxygen levels
6. Bronchoscopy: For severe or atypical cases

X-Ray Interpretation:
- Normal: Clear lung fields, visible lung markings
- Pneumonia indicators: White patches (consolidation), air bronchograms, pleural effusion
- Bacterial: Lobar consolidation pattern
- Viral: Bilateral interstitial infiltrates
- Aspiration: Lower lobe involvement

[SECTION 4: TREATMENT PROTOCOLS]

Bacterial Pneumonia:
- First-line: Amoxicillin, Azithromycin, Doxycycline
- Severe: IV antibiotics (Ceftriaxone + Azithromycin)
- Duration: 5-7 days (mild), 7-14 days (severe)

Viral Pneumonia:
- Supportive care primarily
- Antivirals for influenza (Oseltamivir/Tamiflu)
- COVID-19: Specific protocols as per guidelines

Supportive Care:
- Oxygen therapy if SpO2 < 94%
- IV fluids for dehydration
- Fever management: Acetaminophen, Ibuprofen
- Pain management for pleuritic pain
- Bronchodilators if wheezing present

Home Care Guidelines:
- Rest: Avoid strenuous activity
- Hydration: 8-10 glasses of fluids daily
- Humidity: Use humidifier for comfort
- Positioning: Elevate head while sleeping
- Monitoring: Track temperature, breathing rate

[SECTION 5: PREVENTION STRATEGIES]

Vaccinations:
1. Pneumococcal vaccines:
   - PCV13 (Prevnar 13): For children, adults 65+
   - PPSV23 (Pneumovax 23): Adults 65+, high-risk groups
2. Influenza vaccine: Annual, reduces viral pneumonia risk
3. COVID-19 vaccine: Reduces COVID pneumonia severity

Lifestyle Prevention:
- Hand hygiene: Wash with soap 20+ seconds
- Avoid smoking: Damages lung defenses
- Limit alcohol: Weakens immune response
- Dental hygiene: Reduces aspiration risk
- Avoid sick contacts: Especially for high-risk individuals

High-Risk Groups:
- Adults over 65 years
- Children under 2 years
- Chronic disease patients (COPD, asthma, diabetes, heart disease)
- Immunocompromised individuals
- Smokers and heavy alcohol users

[SECTION 6: RECOVERY INFORMATION]

Typical Recovery Timeline:
- Week 1: Fever resolves, fatigue remains high
- Week 2-3: Cough improves, energy gradually returns
- Week 4-6: Most symptoms resolve
- Week 6-12: Full lung function recovery
- Several months: Complete stamina restoration

Post-Pneumonia Care:
- Follow-up chest X-ray at 6-8 weeks
- Gradual return to activities
- Watch for recurrence signs
- Complete all prescribed medications
- Pulmonary rehabilitation if needed

[SECTION 7: NUTRITION FOR LUNG HEALTH]

Recommended Foods:
- Protein: Lean meats, fish, eggs, legumes (tissue repair)
- Vitamin C: Citrus, berries, bell peppers (immune boost)
- Vitamin D: Fatty fish, fortified foods, sunlight
- Zinc: Shellfish, seeds, nuts, whole grains
- Antioxidants: Colorful vegetables, green tea
- Omega-3: Salmon, walnuts, flaxseed (anti-inflammatory)

Foods to Avoid During Recovery:
- Processed foods with excess sodium
- Sugary drinks and foods
- Alcohol (weakens immunity)
- Dairy if it increases mucus production
- Fried and fatty foods

Hydration:
- Water: 8-10 glasses daily minimum
- Warm fluids: Herbal tea, clear broths
- Electrolyte drinks if feverish
- Avoid caffeine excess (dehydrating)

[SECTION 8: WHEN TO SEEK EMERGENCY CARE]

Call 911 Immediately If:
- Severe difficulty breathing or gasping
- Blue or gray color to lips, face, or nails
- Confusion, disorientation, or unresponsiveness
- Chest pain with sweating or radiating to arm
- Inability to keep fluids down
- Signs of sepsis: rapid heart rate, cold extremities, mottled skin

See Doctor Within 24 Hours If:
- Fever above 102°F (39°C) not responding to medication
- Worsening shortness of breath
- Coughing up blood or rust-colored sputum
- Symptoms improving then suddenly worsening
- New or increasing chest pain
- Persistent vomiting preventing medication intake

[SECTION 9: CHEST X-RAY ANALYSIS CONTEXT]

AI Detection Confidence Levels:
- High Confidence (>85%): Strong indicators present, likely accurate
- Moderate Confidence (65-85%): Suggestive findings, clinical correlation needed
- Low Confidence (<65%): Uncertain findings, additional testing recommended

Limitations of AI X-ray Analysis:
- Cannot determine pneumonia type (bacterial vs viral)
- Cannot assess severity without clinical context
- May miss early or atypical presentations
- Cannot replace clinical examination
- Should be confirmed by radiologist

[SECTION 10: FREQUENTLY ASKED QUESTIONS]

Q: Is pneumonia contagious?
A: Bacterial and viral pneumonia can spread through respiratory droplets. Practice good hygiene and avoid close contact with infected individuals.

Q: Can pneumonia be fatal?
A: Yes, especially in high-risk groups. Early treatment significantly improves outcomes. Seek medical care promptly if symptoms are severe.

Q: How is pneumonia different from bronchitis?
A: Pneumonia affects the lung air sacs (alveoli), while bronchitis affects the bronchial tubes. Pneumonia is generally more severe with higher fever and requires different treatment.

Q: Can you have pneumonia without fever?
A: Yes, especially in elderly patients, immunocompromised individuals, or those on certain medications. Other symptoms may be present.

Q: When can I return to work/school?
A: Generally when fever-free for 24-48 hours without medication, symptoms are improving, and energy levels permit. Consult your doctor for specific guidance.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY CLASSIFICATION AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

ALLOWED_TOPICS = [
    "pneumonia", "lung", "respiratory", "breathing", "chest", "x-ray", "xray",
    "cough", "fever", "infection", "symptoms", "treatment", "diagnosis",
    "recovery", "prevention", "vaccine", "antibiotics", "hospital",
    "oxygen", "mucus", "phlegm", "bronchitis", "asthma", "copd",
    "inflammation", "pulmonary", "health", "doctor", "medical", "medicine",
    "diet", "nutrition", "exercise", "immune", "immunity", "hydration",
    "rest", "sleep", "emergency", "urgent", "care", "risk", "smoking",
    "contagious", "spread", "bacteria", "virus", "viral", "bacterial",
    "test", "scan", "ct", "blood", "sputum", "covid", "flu", "influenza",
    "child", "elderly", "senior", "patient", "sick", "ill", "wellness",
    "pain", "ache", "fatigue", "tired", "weak", "energy", "appetite"
]

def classify_query_with_llm(user_message):
    """
    Use LLM to classify if the query is related to allowed medical topics.
    Returns: (is_relevant: bool, classification: str)
    """
    if not LLM_AVAILABLE or not SAMBANOVA_API_KEY:
        # Fallback to keyword matching if LLM unavailable
        message_lower = user_message.lower()
        for topic in ALLOWED_TOPICS:
            if topic in message_lower:
                return True, "keyword_match"
        return False, "no_match"
    
    classification_prompt = f"""You are a query classifier for a pneumonia detection medical assistant.

Your ONLY task is to determine if the user's question is related to:
1. Pneumonia (symptoms, treatment, prevention, recovery, diagnosis)
2. Respiratory health (lungs, breathing, chest conditions)
3. Chest X-ray analysis and interpretation
4. General health advice related to lung conditions
5. Medical emergencies related to breathing/respiratory issues
6. Diet, nutrition, lifestyle for respiratory health
7. Greetings or general conversation starters (like "hello", "hi", "how are you", "thank you")

User Question: "{user_message}"

Respond with EXACTLY one word:
- "RELEVANT" if the question is about the above topics OR is a greeting/general conversation
- "IRRELEVANT" if the question is about unrelated topics (politics, entertainment, coding, math, other diseases not related to respiratory system, etc.)

Your response (one word only):"""

    try:
        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "Meta-Llama-3.3-70B-Instruct",
                "messages": [{"role": "user", "content": classification_prompt}],
                "temperature": 0.1,
                "max_tokens": 10
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip().upper()
        
        if "RELEVANT" in result:
            return True, "llm_approved"
        else:
            return False, "llm_rejected"
            
    except Exception as e:
        # Fallback to keyword matching on error
        message_lower = user_message.lower()
        for topic in ALLOWED_TOPICS:
            if topic in message_lower:
                return True, "keyword_fallback"
        return False, "classification_error"


def get_off_topic_warning():
    """Return warning message for off-topic queries"""
    return """⚠️ **Off-Topic Query Detected**

I'm a specialized **Pneumonia Detection Medical Assistant** trained specifically for:

🫁 **My Expertise Areas:**
- Pneumonia symptoms, causes, and types
- Chest X-ray analysis and interpretation
- Treatment options and recovery guidance
- Prevention strategies and vaccinations
- Respiratory health and lung conditions
- Diet and nutrition for lung health
- When to seek emergency care

❌ **I cannot help with:**
- Topics unrelated to respiratory/lung health
- Non-medical questions (coding, math, entertainment, etc.)
- Other medical conditions outside my training

💡 **Try asking me:**
- "What are the symptoms of pneumonia?"
- "How do I interpret my X-ray results?"
- "What foods help with recovery?"
- "When should I go to the emergency room?"
- "How can I prevent pneumonia?"

---
*I'm designed to provide accurate, focused medical information within my domain.*"""

# ═══════════════════════════════════════════════════════════════════════════════
# SMART MEDICAL ASSISTANT WITH TRAINED LLM
# ═══════════════════════════════════════════════════════════════════════════════

def ask_smart_assistant(user_message, pred_result=None, confidence=None):
    """
    Process user query through trained LLM with domain validation.
    Uses comprehensive medical knowledge base for context.
    """
    
    # Step 1: Check if LLM is available
    if not LLM_AVAILABLE or not SAMBANOVA_API_KEY:
        return """⚠️ **AI Assistant Unavailable**

The AI Medical Assistant requires a SambaNova API key to function.

**To enable the assistant:**
1. Get your API key from: https://cloud.sambanova.ai/apis
2. Add it to your environment: `SAMBANOVA_API_KEY=your_key_here`
3. For Hugging Face: Add to Space Secrets

---
*The pneumonia detection feature still works without the API key.*"""
    
    # Step 2: Classify the query using LLM
    is_relevant, classification = classify_query_with_llm(user_message)
    
    # Step 3: Return warning for off-topic queries
    if not is_relevant:
        return get_off_topic_warning()
    
    # Step 4: Build comprehensive system prompt with training data
    system_prompt = f"""You are a SPECIALIZED Medical Information Assistant for Pneumonia Detection.

=== CRITICAL INSTRUCTIONS ===
1. You MUST ONLY answer questions related to:
   - Pneumonia (all types, symptoms, treatment, prevention, recovery)
   - Respiratory health and lung conditions
   - Chest X-ray interpretation and analysis
   - Related medical emergencies
   - Diet and lifestyle for respiratory health

2. You MUST NOT:
   - Provide specific diagnoses (always recommend consulting a doctor)
   - Prescribe medications with dosages
   - Answer questions outside your domain
   - Make up information not in your training
   - Give advice that could delay emergency care

3. Response Guidelines:
   - Be concise (under 300 words)
   - Use clear formatting with emojis
   - Include relevant medical disclaimers
   - Recommend professional consultation when appropriate
   - Base answers ONLY on the provided medical knowledge

=== YOUR MEDICAL KNOWLEDGE BASE ===
{MEDICAL_TRAINING_CONTEXT}

=== END OF KNOWLEDGE BASE ===

Remember: You are an INFORMATIONAL assistant, NOT a replacement for medical professionals.
Always encourage users to seek proper medical care for their conditions."""

    # Step 5: Add X-ray analysis context if available
    xray_context = ""
    if pred_result and confidence:
        severity = "high" if confidence > 85 else "moderate" if confidence > 65 else "low"
        xray_context = f"""

=== CURRENT X-RAY ANALYSIS CONTEXT ===
The user has uploaded a chest X-ray with the following AI analysis:
- Prediction: {pred_result}
- Confidence: {confidence:.1f}%
- Confidence Level: {severity.upper()}

IMPORTANT: This is an AI estimation for educational purposes.
The AI model has {confidence:.1f}% confidence in detecting {'pneumonia indicators' if pred_result == 'PNEUMONIA' else 'normal lung patterns'}.
Always recommend professional radiologist review and clinical correlation.
=== END X-RAY CONTEXT ==="""

    # Step 6: Prepare messages for LLM
    messages = [
        {"role": "system", "content": system_prompt + xray_context},
        {"role": "user", "content": user_message}
    ]
    
    # Step 7: Call LLM API
    try:
        with st.spinner("🤔 Analyzing your question..."):
            response = requests.post(
                "https://api.sambanova.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Meta-Llama-3.3-70B-Instruct",
                    "messages": messages,
                    "temperature": 0.3,  # Lower temperature for more factual responses
                    "max_tokens": 500,
                    "top_p": 0.9
                },
                timeout=30
            )
            response.raise_for_status()
            
            llm_response = response.json()["choices"][0]["message"]["content"]
            
            # Add source attribution
            return llm_response + "\n\n---\n*🤖 Response generated by trained Medical AI | Always consult healthcare professionals*"
            
    except requests.exceptions.Timeout:
        return """⚠️ **Request Timeout**

The AI is taking longer than expected. Please try again.

If the problem persists, check your internet connection."""
        
    except requests.exceptions.RequestException as e:
        return f"""⚠️ **Connection Error**

Could not connect to the AI service.

**Error:** {str(e)}

**Troubleshooting:**
1. Check your internet connection
2. Verify your API key is correct
3. Try again in a few moments"""
        
    except Exception as e:
        return f"""⚠️ **Unexpected Error**

An error occurred while processing your request.

**Error:** {str(e)}

Please try rephrasing your question or try again later."""


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
    st.title("🫁 Pneumonia Detection")
    st.markdown("### AI-Powered Medical Image Analysis")
    

    with st.sidebar:
        st.header("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card"><h3 style="margin:0; color:#1f77b4;">95.2%</h3><p style="margin:0; font-size:14px;">AUC-ROC</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3 style="margin:0; color:#2ca02c;">87%</h3><p style="margin:0; font-size:14px;">Accuracy</p></div>', unsafe_allow_html=True)
        
        st.header("🔧 Technology")
        st.info("**Detection:** TensorFlow CNN\n**Chat:**Recommendations:** Expert system")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Chest X-Ray")
        uploaded_file = st.file_uploader("Choose X-ray (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded X-Ray", width="stretch")

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
            if st.button(label, use_container_width=True):  # Note: use_container_width for buttons is still valid
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
        ai_status = "🤖 Trained Medical AI" if LLM_AVAILABLE and SAMBANOVA_API_KEY else "⚠️ API Key Required"
        st.session_state.messages = [
            {"role": "assistant", "content": f"""👋 **Hello! I'm your Trained Pneumonia Medical Assistant!**

{ai_status}

🎯 **My Specialized Training Includes:**
- 📊 Chest X-ray analysis interpretation
- 🦠 Pneumonia types, symptoms & severity levels
- 💊 Treatment protocols & recovery guidance
- 🛡️ Prevention strategies & vaccinations
- 🥗 Nutrition for respiratory health
- 🚨 Emergency warning signs

⚠️ **Important:** I only answer questions within my medical training domain. Off-topic questions will receive a warning.

💬 **Ask me anything about pneumonia or respiratory health!**"""}
        ]

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Use text input instead of chat_input (works inside tabs)
    user_input = st.text_input("💬 Ask about pneumonia or your X-ray...", key="chat_text_input", placeholder="Type your question here...")
    
    if st.button("Send Message", type="primary", use_container_width=True):
        if user_input and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})

            response = ask_smart_assistant(
                user_input,
                st.session_state.pred_result if st.session_state.analysis_done else None,
                st.session_state.confidence if st.session_state.analysis_done else None
            )

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Footer
st.markdown("---")

st.warning("🫁 Pneumonia Detection AI can make mistakes — Consult healthcare professionals")
