import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_curve,roc_auc_score,roc_curve
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
#load model
model=tf.keras.models.load_model('models/pneumonia_model.h5')
#prepare test data
test_datagen=ImageDataGenerator(rescale=1./255)
test_generator=test_datagen.flow_from_directory(
    'data/chest_xray/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False
)

# Predictions
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Classification report
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('models/confusion_matrix.png')
plt.show()
predictions=model.predict(test_generator)
auc_score=roc_auc_score(y_true,predictions)
print(f"Auc-roc score:{auc_score:.4f}")
fpr,tpr,thresholds=roc_curve(y_true,predictions)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f'AUC={auc_score:.3f}')
plt.plot([0,1],[0,1],'k',label='Random')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('roc curve')
plt.legend()
plt.savefig('models/roc_curve.png')
plt.show()

