import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow is not installed. Please check your requirements.txt file.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .normal {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .pneumonia {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/pneumonia_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure the model file exists at 'models/pneumonia_model.h5'")
        st.info("If deploying on Streamlit Cloud, ensure all required files are uploaded")
        return None

# Preprocess image
def preprocess_image(image):
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 224, 224, 1)
    return img_array, img

# Make prediction
def predict(model, image):
    processed_img, display_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)[0][0]
    return prediction, display_img

# Main app
def main():
    # Header
    st.title("🫁 Pneumonia Detection from Chest X-Ray")
    st.markdown("### AI-Powered Medical Image Analysis")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info(
            "This application uses a Convolutional Neural Network (CNN) "
            "to detect pneumonia from chest X-ray images."
        )
        
        st.header("📊 Model Information")
        st.markdown("""
        - **Architecture**: Custom CNN (Sequential)
        - **Input Size**: 224x224 pixels (Grayscale)
        - **Classes**: Normal / Pneumonia
        - **Framework**: TensorFlow/Keras
        - **Training**: Enhanced with data augmentation & class weighting
        """)
        
        st.header("🎯 How to Use")
        st.markdown("""
        1. Upload a chest X-ray image
        2. Click 'Analyze X-Ray'
        3. View the prediction results
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning(
            "This tool is for educational purposes only. "
            "Always consult healthcare professionals for medical diagnosis."
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)

            # Show processed image
            processed_img_array, processed_img = preprocess_image(image)
            st.image(processed_img, caption="Processed X-Ray (Grayscale)", use_container_width=True)

            # Analyze button
            analyze_button = st.button("🔍 Analyze X-Ray")
            
            if analyze_button:
                model = load_model()
                
                if model is not None:
                    with st.spinner("Analyzing image..."):
                        try:
                            prediction, processed_img = predict(model, image)
                            
                            # Store results in session state
                            st.session_state.prediction = prediction
                            st.session_state.processed_img = processed_img
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state.prediction
            
            # Determine result
            if prediction > 0.5:
                result = "PNEUMONIA"
                confidence = prediction * 100
                box_class = "pneumonia"
                icon = "🔴"
            else:
                result = "NORMAL"
                confidence = (1 - prediction) * 100
                box_class = "normal"
                icon = "🟢"
            
            # Display result
            st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2>{icon} Prediction: {result}</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence meter
            st.subheader("Confidence Breakdown")
            
            normal_conf = (1 - prediction) * 100
            pneumonia_conf = prediction * 100
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Normal", f"{normal_conf:.2f}%")
            with col_b:
                st.metric("Pneumonia", f"{pneumonia_conf:.2f}%")
            
            # Progress bars
            st.progress(float(normal_conf / 100), text=f"Normal: {normal_conf:.1f}%")
            st.progress(float(pneumonia_conf / 100), text=f"Pneumonia: {pneumonia_conf:.1f}%")
            
            # Visualization
            st.subheader("📈 Confidence Visualization")
            fig, ax = plt.subplots(figsize=(8, 5))
            
            categories = ['Normal', 'Pneumonia']
            confidences = [normal_conf, pneumonia_conf]
            colors = ['#28a745' if normal_conf > pneumonia_conf else '#6c757d',
                     '#dc3545' if pneumonia_conf > normal_conf else '#6c757d']
            
            bars = ax.barh(categories, confidences, color=colors)
            ax.set_xlabel('Confidence (%)', fontsize=12)
            ax.set_xlim(0, 100)
            ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                ax.text(conf + 2, i, f'{conf:.1f}%', 
                       va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretation
            st.subheader("🔬 Interpretation")
            if result == "PNEUMONIA":
                if confidence > 90:
                    st.error("⚠️ High confidence of pneumonia detected. Immediate medical consultation recommended.")
                elif confidence > 70:
                    st.warning("⚠️ Moderate to high confidence of pneumonia. Please consult a healthcare professional.")
                else:
                    st.info("ℹ️ Possible pneumonia detected. Further examination recommended.")
            else:
                if confidence > 90:
                    st.success("✅ High confidence that the X-ray appears normal.")
                elif confidence > 70:
                    st.success("✅ X-ray appears normal with good confidence.")
                else:
                    st.info("ℹ️ X-ray likely normal, but additional tests may be beneficial.")
        
        else:
            st.info("👆 Upload an X-ray image and click 'Analyze' to see results")
            
            # Sample images info
            st.markdown("---")
            st.subheader("💡 Need a sample?")
            st.markdown("""
            You can test this application with chest X-ray images from:
            - Your test dataset
            - Medical imaging databases (with proper permissions)
            - The Kaggle Chest X-Ray dataset
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Developed with ❤️ using TensorFlow and Streamlit</p>
            <p>⚕️ For Educational and Research Purposes Only</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()