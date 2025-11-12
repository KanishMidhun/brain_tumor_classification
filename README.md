🧠 Brain Tumor MRI Classification — Deep Learning & Streamlit App
An end-to-end deep learning project that classifies Brain MRI images into different tumor categories using Custom CNN, EfficientNetB0, and ResNet50.
Models are trained in Google Colab and deployed with a Streamlit Web App (app1.py) for real-time predictions.

📁 Project Overview
This project demonstrates how deep learning can be applied to medical imaging for tumor detection and classification.
It includes:
Model training and evaluation in Google Colab
Model export and deployment to a Streamlit web interface

🧩 Dataset Details
The dataset consists of MRI images categorized into four classes:
🟢 Glioma Tumor
🔵 Meningioma Tumor
🟣 Pituitary Tumor
⚪ No Tumor

Folder structure:
Brain_Tumor_Dataset/
│
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
└── val/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/

⚙️ Model Training Workflow

The entire training process is inside the notebook
📘 Brain_Tumor_MRI_Classification_Colab_Notebook.ipynb

📌 Step-by-step pipeline:

1️⃣ Understand the Dataset
Checked image distribution and class balance
Ensured consistent resolution and folder structure

2️⃣ Data Preprocessing
Resized to (224x224)
Normalized pixel values to [0–1]

3️⃣ Data Augmentation
Applied to improve generalization:
Rotation, flip, zoom, brightness shift, and translation
4️⃣ Model Building

Implemented three architectures:
🧠 Custom CNN (built from scratch)
⚡ EfficientNetB0 (transfer learning from ImageNet)
🧱 ResNet50 (transfer learning from ImageNet)
5️⃣ Model Training

Optimizer: Adam
Loss: categorical_crossentropy
Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
Trained for 10–25 epochs
6️⃣ Model Evaluation

Metrics:
Accuracy, Precision, Recall, F1-score
Confusion Matrix visualized for each model

🧠 Exporting Models
After training in Colab, all the models and label files were exported for deployment:
exports/
├── best_custom_cnn.keras
├── best_efficientnetb0.keras
├── best_resnet50.keras
└── class_names.json
These files are used by the Streamlit app.

💻 Streamlit App (app1.py)

This app allows users to:
Upload an MRI image.
Choose which model to use (EfficientNetB0 / ResNet50 / Custom CNN).
Get instant tumor classification with confidence visualization.

🧩 Features:
✅ Multiple model selection
✅ Real-time prediction
✅ Probability bar chart
✅ Fast cached loading
✅ Lightweight UI built with Streamlit


