# 🫁 Pneumonia Detection AI + Smart Assistant

An advanced AI-powered medical image analysis system that combines Convolutional Neural Networks (CNN) for pneumonia detection with a sophisticated Smart Medical Assistant powered by SambaNova Cloud AI. Features include real-time X-ray analysis, Top 5 personalized recommendations, and an intelligent chat assistant for medical queries. Built with Streamlit for an intuitive web interface.

##  Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web Application](#web-application)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

##  Features

- **Deep Learning Model**: CNN-based pneumonia detection using TensorFlow/Keras
- **Smart Medical Assistant**: AI-powered chat assistant using SambaNova Cloud (Meta-Llama-3.3-70B) for medical queries
- **Top 5 Recommendations**: Personalized expert recommendations based on X-ray analysis results
- **Web Interface**: User-friendly Streamlit application with dual-tab interface for analysis and consultation
- **Data Augmentation**: Enhanced training with image preprocessing techniques
- **Model Evaluation**: Comprehensive evaluation metrics and confusion matrix
- **Visualization**: Training history and prediction confidence charts
- **Real-time Prediction**: Instant analysis of uploaded X-ray images
- **Built-in Knowledge Base**: Fallback medical information for common questions
- **Confidence-based Recommendations**: Dynamic suggestions based on AI prediction confidence levels

##  Installation

### Prerequisites

- Python 3.11.0
- TensorFlow 2.15.0
- Streamlit
- Required Python packages (see requirements.txt)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd pneumonia-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the project root
   - Add your SambaNova API key: `SAMBANOVA_API_KEY=your_key_here`
   - Get your API key from https://cloud.sambanova.ai/apis

5. **Download the dataset:**
   - Download the Chest X-Ray Images (Pneumonia) dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
   - Extract the dataset to the `data/chest_xray/` directory

##  Usage

### Training the Model

Run the training script to build and train the CNN model:

```bash
python src/train.py
```

This will:
- Load and preprocess the training data
- Train the CNN model with data augmentation
- Save the trained model to `models/pneumonia_model.h5`
- Generate training history plots

### Evaluating the Model

Evaluate the trained model on the test dataset:

```bash
python src/evaluate.py
```

This generates:
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization

### Making Predictions

Use the prediction script for individual image analysis:

```bash
python src/predict.py --image path/to/xray/image.jpg
```

### Web Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` to use the application.

##  Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset, which contains:

- **Training set**: 5,216 images (Normal: 1,341, Pneumonia: 3,875)
- **Validation set**: 16 images (Normal: 8, Pneumonia: 8)
- **Test set**: 624 images (Normal: 234, Pneumonia: 390)

### Data Structure

```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

##  Model Architecture

The CNN model consists of:

- **Input Layer**: 224x224x1 grayscale images
- **Convolutional Layers**: 4 Conv2D layers with BatchNormalization and increasing filters (32, 64, 128, 128)
- **Pooling Layers**: MaxPooling2D layers for downsampling
- **Regularization**: Dropout layers (0.5 after Flatten, 0.3 before Dense) to prevent overfitting
- **Dense Layers**: 512 neurons with ReLU activation
- **Output Layer**: Single neuron with sigmoid activation for binary classification

### Model Summary

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 222, 222, 32)      320
conv2d_1 (Conv2D)            (None, 220, 220, 64)      18496
conv2d_2 (Conv2D)            (None, 218, 218, 128)     73856
conv2d_3 (Conv2D)            (None, 216, 216, 128)     147584
flatten (Flatten)            (None, 5668352)           0
dropout (Dropout)            (None, 5668352)           0
dense (Dense)                (None, 512)               2900606976
dense_1 (Dense)              (None, 1)                 513
=================================================================
Total params: 2,902,646,745
Trainable params: 2,902,646,745
Non-trainable params: 0
```

##  Training

### Hyperparameters

- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 15 (with early stopping)
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall

### Advanced Features

- **Early Stopping**: Monitors validation loss with patience of 5 epochs
- **Model Checkpointing**: Saves the best model based on validation accuracy
- **Class Weighting**: Handles imbalanced dataset by computing balanced class weights
- **Regularization**: BatchNormalization and Dropout layers

### Data Augmentation

- Random rotation (±20°)
- Width and height shift (±20%)
- Shear transformation (±15°)
- Zoom range (±15%)
- Horizontal flipping
- Rescaling to [0,1] range

##  Evaluation

The model is evaluated using comprehensive classification metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve

### Sample Results

```
              precision    recall  f1-score   support

      Normal       0.89      0.75      0.81       234
   Pneumonia       0.87      0.95      0.91       390

    accuracy                           0.88       624
   macro avg       0.88      0.85      0.86       624
weighted avg       0.88      0.88      0.87       624
```

##  Web Application
The Streamlit application provides:

- **Image Upload**: Support for PNG, JPG, JPEG formats
- **Real-time Analysis**: Instant prediction with confidence scores
- **Visual Results**: Interactive Plotly charts and progress bars
- **Confidence Visualization**: Bar charts showing prediction probabilities
- **Medical Disclaimer**: Clear warnings about educational use only

### Interface Features

- Drag-and-drop image upload with image preview
- Side panel with model information and usage instructions
- Color-coded prediction results (green for normal, red for pneumonia)
- Confidence meters and progress bars for both classes
- Interactive Plotly bar charts for confidence visualization
- Interpretation section with medical recommendations
- Custom CSS styling for enhanced user experience
- Responsive design for different screen sizes
- Enhanced error handling and loading states

##  Results

### Training History

![Training History](models/training_history.png)

### Confusion Matrix

![Confusion Matrix](models/confusion_matrix.png)

### ROC Curve

![ROC Curve](models/roc_curve.png)

### Performance Metrics

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~90%
- **Test Accuracy**: ~88%
- **Precision (Pneumonia)**: 87%
- **Recall (Pneumonia)**: 95%
- **AUC-ROC**: ~0.93

##  Project Structure

```
pneumonia-prediction/
├── app.py                      # Streamlit web application
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python runtime version
├── data/
│   └── chest_xray/            # Dataset directory
│       ├── train/
│       ├── val/
│       └── test/
├── models/                     # Trained models and outputs
│   ├                           # Trained CNN model
│   ├── pneumonia_model_best.h5 # Best model checkpoint
│   ├── training_history.png    # Training visualization
│   ├── confusion_matrix.png    # Evaluation results
│   ├── roc_curve.png          # ROC curve visualization
│   └── notebook/
│       └── eda.ipynb          # Exploratory data analysis
└── src/                       # Source code
    ├── __init__.py
    ├── train.py               # Model training script
    ├── evaluate.py            # Model evaluation script
    └── predict.py             # Prediction script
```

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Write unit tests for new features
- Update documentation for API changes

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Disclaimer

**This application is for educational and research purposes only.**

- **Not for clinical use**: This tool should not be used for actual medical diagnosis or treatment decisions.
- **Consult professionals**: Always consult qualified healthcare professionals for medical advice and diagnosis.
- **No liability**: The developers and contributors are not responsible for any consequences arising from the use of this software.
- **Data privacy**: Ensure compliance with relevant data protection regulations when handling medical images.

##  Acknowledgments

- **Dataset**: Chest X-Ray Images (Pneumonia) from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **Medical Community**: For providing the dataset and research foundation

##  Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.

---

**Developed with  for advancing medical AI research**
