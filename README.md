# 🖼️ E-Waste Image Classification using EfficientNet

This project is a deep learning-based image classification system that identifies types of electronic waste (e-waste) using a custom-trained model based on EfficientNetV2. It helps in automating the classification process of e-waste images for better recycling and environmental impact analysis.

## 🚀 Project Overview

Electronic waste is one of the fastest-growing pollution problems worldwide. Manual sorting is time-consuming and error-prone. This project uses deep learning to classify e-waste images into categories such as:

- Mobile Phones
- Keyboards
- Monitors
- PCBs
- Batteries
- Televisions
- ...and more (up to 10 classes)

## 🧠 Model

We used the **EfficientNetV2B0** and **EfficientNetV2B2** architectures due to their performance efficiency on image classification tasks.

### Features:
- Transfer learning with pre-trained EfficientNet weights
- Trained with real-world e-waste images
- Achieved up to **96.67% validation accuracy**

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetV2
- Matplotlib, NumPy, Pandas
- Google Colab for training

## 📊 Dataset

The dataset contains labeled images of various e-waste categories. Images are resized to 224x224 and split into training and validation sets.

> Note: Dataset is not included due to licensing; contact the author or replace with your own labeled dataset.

## 🧪 Training Summary

```python
checkpoint = ModelCheckpoint("ewaste_classification_prakriti.keras", save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
