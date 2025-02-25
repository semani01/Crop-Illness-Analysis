# FloraSense: Deep Learning for Crop Illness Analysis 🌿

<p align="center">
  <!-- Badge linking to your LICENSE file in GitHub -->
  <a href="https://github.com/semani01/Crop-Illness-Analysis/blob/main/LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-darkgreen.svg" />
  </a>
</p>

## 🌟 Overview
FloraSense is an advanced **deep learning-based solution** designed to **detect and classify plant diseases** using **image classification techniques**. The project leverages **hybrid Convolutional Neural Networks (CNNs)** by combining **ResNet50** and **InceptionV3** to achieve **high accuracy in plant disease identification**.

### 🚀 Key Features
- **Automated Disease Detection**: Identifies plant diseases from leaf images.
- **Hybrid CNN Model**: Uses **ResNet50 + InceptionV3** for enhanced feature extraction.
- **Data Augmentation**: Improves model generalization with rotation, zoom, and flip transformations.
- **High Accuracy**: Achieves significant accuracy in multi-class classification of plant diseases.
- **Scalability**: Can be extended for real-world agricultural applications.

---

## 📌 Problem Statement
### Why is Plant Disease Detection Important?
- **Early detection** is crucial to prevent major crop losses.
- Traditional **manual inspection** methods are **slow, expensive, and prone to errors**.
- **Automated deep learning-based detection** can improve **agricultural efficiency** and support **farmers with timely intervention**.
- **Sustainability**: Helps reduce excessive pesticide usage by ensuring targeted treatment.

---

## 🎯 Motivation
### Why This Project?
- 🌍 **Food Security**: Aims to reduce crop loss and enhance **global food production**.
- 🤖 **Technological Advancement**: Integrates **deep learning and computer vision** to improve disease detection accuracy.
- ♻ **Sustainability**: Reduces pesticide misuse and contributes to **eco-friendly farming practices**.

---

## 📊 Dataset Details
- **Total Images**: ~90,000 images
- **Classes**: 38 categories (including healthy and diseased plant leaves)
- [Download the dataset here](https://drive.google.com/drive/u/1/folders/17bSmjnpMOIEhUnfhuoWIQEbaj80SbY3A)
- **Example Labels**:
  - 🍎 `Apple Scab`
  - 🍅 `Tomato Early Blight`
  - 🥔 `Potato Late Blight`
  - ✅ `Healthy Plants`
- **Dataset Preprocessing**:
  - All images resized to **256x256 pixels**.
  - Data augmentation applied for improved model robustness.

---

## 🔬 Model Architecture
### Hybrid CNN Approach
To achieve optimal performance, FloraSense combines two **powerful pre-trained CNN models**:

- **ResNet50**: Extracts **high-level spatial features**.
- **InceptionV3**: Captures **fine-grained texture details**.

### Model Components
- **Feature Extraction**: Uses ResNet50 and InceptionV3 without their top layers.
- **Flatten & Concatenation**: Merges feature maps from both models.
- **Dense Layers**: Fully connected layers for classification.
- **Dropout (0.3)**: Prevents overfitting.
- **Softmax Activation**: Classifies the input into **38 plant disease categories**.
- **Optimizer**: Adam (learning rate adjusted dynamically).
- **Loss Function**: Categorical Crossentropy.

---

## ⚡ Training and Evaluation
### Training Strategy
- **Data Augmentation**: Improves generalization using:
  - **Rotation** (up to 40 degrees)
  - **Zooming & Shifting**
  - **Flipping** (horizontal & vertical)
- **Batch Size**: 32
- **Epochs**: 50 (early stopping applied)
- **GPU Acceleration**: Training performed using **NVIDIA GPU**

### Evaluation Metrics
- **Accuracy**: Measures correct classifications.
- **Loss Curve Analysis**: Evaluates convergence and generalization.
- **Confusion Matrix**: Identifies misclassified plant diseases.
- **Precision, Recall, F1-Score**: Measures model robustness.

---

## 📈 Results & Insights
- **Validation Accuracy**: High accuracy achieved (>92%)
- **Key Observations**:
  - **Common diseases** were detected with high confidence.
  - **Rare diseases** had lower accuracy due to limited data.
  - **Data augmentation significantly improved generalization**.

---

## 🛠️ Installation & Usage
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/FloraSense-Plant-Disease-Detection.git
cd FloraSense-Plant-Disease-Detection
```
### 2️⃣ Open the Jupyter Notebook
- Open VS Code or Jupyter Notebook.
- Run the DL_Project.ipynb file.
- Ensure the dataset is available in the correct path.

### 3️⃣ Train or Test the Model
- To train the model:
```bash
model.fit(train_generator, epochs=50, validation_data=validation_generator)
```
- To test on a new image:
```bash
img = load_img('path_to_image.jpg', target_size=(256, 256))
prediction = model.predict(img)
print("Predicted Disease Class:", prediction)
```

---

## 🚀 Future Enhancements
- **Real-Time Mobile App**: Deploy the model on edge devices for real-time farm use.
- **Environmental Data Integration**: Incorporate humidity, temperature, soil quality.
- **More Efficient CNN Models**: Optimize model size for faster inference.
- **Ethical AI**: Ensure fairness and unbiased classification across all plant species.

---

## 👥 Contributors
- Sai Srikar Emani (saisrikar.emani@ucdenver.edu)

---

## License
This project is licensed under the **MIT License**. Feel free to modify and enhance!

🌟 If you find this project useful, don't forget to ⭐ it on GitHub!

