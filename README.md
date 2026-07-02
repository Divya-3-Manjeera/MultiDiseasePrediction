# 🏥 Multi-Disease Prediction System

An AI-powered web application that predicts multiple diseases using machine learning and deep learning models. Built with **Flask**, **Scikit-learn**, and **TensorFlow**, the system provides an intuitive interface for early disease detection and health assessment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Disease Modules](#disease-modules)
  - [1. Diabetes Prediction](#1-diabetes-prediction)
  - [2. Kidney Stone Detection](#2-kidney-stone-detection)
  - [3. Symptom-Based Disease Prediction](#3-symptom-based-disease-prediction)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Models Used](#models-used)
- [Datasets](#datasets)
- [Future Enhancements](#future-enhancements)

---

## 🔍 Overview

The **Multi-Disease Prediction System** is a full-stack web application designed to assist users in early disease detection. It integrates three independent prediction modules — **Diabetes Prediction**, **Kidney Stone Detection**, and **Symptom-Based Disease Prediction** — into a single unified platform. Each module leverages trained ML/DL models to provide real-time predictions based on user inputs.

---

## ✨ Features

- 🩺 **Three Disease Prediction Modules** in a single platform
- 🚻 **Gender-Specific Diabetes Models** — separate models for male and female patients
- 🖼️ **CT Scan Image Upload** for kidney stone detection using deep learning
- 💊 **Comprehensive Health Recommendations** — medications, precautions, diet, and workout suggestions for symptom-based predictions
- ✅ **Input Validation** — all form fields have clinically valid min/max ranges with placeholder hints
- 📱 **Responsive UI** — built with Bootstrap 5 for a clean, modern look
- ⚡ **Real-Time Predictions** — instant results powered by pre-trained models

---

## 🛠️ Tech Stack

| Layer        | Technology                                         |
| ------------ | -------------------------------------------------- |
| **Backend**  | Python, Flask                                      |
| **Frontend** | HTML5, CSS3, Bootstrap 5, Jinja2 Templates         |
| **ML Models**| Scikit-learn (SVC, Logistic Regression)             |
| **DL Model** | TensorFlow / Keras (CNN for image classification)  |
| **Data**     | Pandas, NumPy                                      |

---

## 📂 Project Structure

```
Multi-Disease Prediction System/
│
├── main.py                          # Flask application (all routes & logic)
│
├── models/
│   ├── symptoms_model.pkl           # SVC model for symptom-based prediction
│   ├── male_diabetes_model.pkl      # Logistic Regression for male diabetes
│   ├── female_diabetes_model.pkl    # ML model for female diabetes (Pima dataset)
│   ├── kidney_model.keras           # CNN model for kidney stone detection
│   ├── heart_rf_model.pkl           # Random Forest model for heart disease (future)
│   ├── heart_scaler.pkl             # Scaler for heart model (future)
│   └── heart_features.pkl           # Feature names for heart model (future)
│
├── datasets/
│   ├── Training.csv                 # Training data for symptom model
│   ├── Symptom-severity.csv         # Symptom severity scores
│   ├── symtoms_df.csv               # Symptom-disease mapping
│   ├── description.csv              # Disease descriptions
│   ├── precautions_df.csv           # Disease precautions
│   ├── medications.csv              # Recommended medications
│   ├── diets.csv                    # Recommended diets
│   └── workout_df.csv               # Recommended workouts
│
├── templates/
│   ├── home.html                    # Landing page with module cards
│   ├── diabetic_pred.html           # Diabetes prediction form (male/female)
│   ├── kidney_pred.html             # Kidney stone CT scan upload
│   └── symptoms_pred.html           # Symptom checker with results
│
├── static/
│   ├── dia_wp.jpeg                  # Diabetes card image
│   ├── kid_wp.jpeg                  # Kidney card image
│   ├── sym_wp.jpeg                  # Symptom card image
│   └── uploads/                     # Uploaded CT scan images
│
├── venv/                            # Python virtual environment
└── README.md                        # Project documentation (this file)
```

---

## 🧬 Disease Modules

### 1. Diabetes Prediction

Provides **gender-specific** diabetes risk assessment using separate ML models.

#### 🔵 Male Form

| Parameter        | Input Type | Min   | Max   | Description                          |
| ---------------- | ---------- | ----- | ----- | ------------------------------------ |
| Age              | Number     | 1     | 120   | Patient's age in years               |
| Hypertension     | Select     | 0 / 1 | —     | 0 = No, 1 = Yes                      |
| Heart Disease    | Select     | 0 / 1 | —     | 0 = No, 1 = Yes                      |
| Smoking History  | Select     | —     | —     | Never / Current / Former / Ever / Not Current / No Info |
| BMI              | Decimal    | 10.0  | 60.0  | Body Mass Index                      |
| HbA1c Level      | Decimal    | 3.5   | 15.0  | Glycated hemoglobin level            |
| Blood Glucose    | Number     | 50    | 500   | Blood glucose level (mg/dL)          |

> **Model:** Logistic Regression with one-hot encoded smoking history (12 features total)

#### 🔴 Female Form (Pima Indians Diabetes Dataset)

| Parameter          | Input Type | Min   | Max   | Description                              |
| ------------------ | ---------- | ----- | ----- | ---------------------------------------- |
| Pregnancies        | Number     | 0     | 20    | Number of times pregnant                 |
| Glucose            | Number     | 40    | 500   | Plasma glucose concentration (mg/dL)     |
| Blood Pressure     | Number     | 40    | 200   | Diastolic blood pressure (mm Hg)         |
| Skin Thickness     | Number     | 0     | 100   | Triceps skin fold thickness (mm)         |
| Insulin            | Number     | 0     | 900   | 2-hour serum insulin (mu U/ml)           |
| BMI                | Decimal    | 10.0  | 70.0  | Body Mass Index                          |
| Diabetes Pedigree  | Decimal    | 0.05  | 2.50  | Diabetes pedigree function score         |
| Age                | Number     | 1     | 120   | Patient's age in years                   |

> **Model:** ML classifier trained on the Pima Indians Diabetes Dataset (8 features)

---

### 2. Kidney Stone Detection

Uses a **Convolutional Neural Network (CNN)** to detect kidney stones from CT scan images.

| Input          | Description                                      |
| -------------- | ------------------------------------------------ |
| CT Scan Image  | Upload a kidney CT scan image (JPEG/PNG)         |

| Output         | Description                                      |
| -------------- | ------------------------------------------------ |
| Stone Detected | Kidney stone is present in the scan              |
| Normal         | No kidney stone detected                         |

> **Model:** TensorFlow/Keras CNN (`kidney_model.keras`) — Image size: 224×224, Binary classification with 0.5 threshold

---

### 3. Symptom-Based Disease Prediction

Enter comma-separated symptoms and the system predicts the most likely disease from **41 possible conditions**, along with:

- 📖 **Disease Description**
- 🛡️ **Precautions** (4 precautionary measures)
- 💊 **Medications**
- 🥗 **Recommended Diet**
- 🏋️ **Workout Suggestions**

> **Model:** Support Vector Classifier (SVC) trained on 132 symptoms across 41 diseases

#### Supported Diseases (41)

| | | | |
|---|---|---|---|
| Fungal Infection | Allergy | GERD | Chronic Cholestasis |
| Drug Reaction | Peptic Ulcer | AIDS | Diabetes |
| Gastroenteritis | Bronchial Asthma | Hypertension | Migraine |
| Cervical Spondylosis | Paralysis | Jaundice | Malaria |
| Chicken Pox | Dengue | Typhoid | Hepatitis A–E |
| Alcoholic Hepatitis | Tuberculosis | Common Cold | Pneumonia |
| Piles | Heart Attack | Varicose Veins | Hypothyroidism |
| Hyperthyroidism | Hypoglycemia | Osteoarthritis | Arthritis |
| Vertigo | Acne | UTI | Psoriasis |
| Impetigo | | | |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Multi-Disease-Prediction-System.git
   cd Multi-Disease-Prediction-System
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install flask pandas numpy scikit-learn tensorflow werkzeug
   ```

5. **Ensure all model files are present in the `models/` directory**
   - `symptoms_model.pkl`
   - `male_diabetes_model.pkl`
   - `female_diabetes_model.pkl`
   - `kidney_model.keras`

6. **Ensure all dataset files are present in the `datasets/` directory**
   - `symtoms_df.csv`, `description.csv`, `precautions_df.csv`
   - `medications.csv`, `diets.csv`, `workout_df.csv`

---

## 🚀 How to Run

```bash
python main.py
```

The application will start on **http://127.0.0.1:5000/**

### Available Routes

| Route              | Method     | Description                            |
| ------------------ | ---------- | -------------------------------------- |
| `/`                | GET        | Home page with all module cards        |
| `/diabetic_pred`   | GET        | Diabetes prediction page               |
| `/predict_male`    | POST       | Male diabetes prediction endpoint      |
| `/predict_female`  | POST       | Female diabetes prediction endpoint    |
| `/kidney_pred`     | GET, POST  | Kidney stone detection page            |
| `/predict`         | GET, POST  | Symptom-based disease prediction page  |

---

## 🧠 Models Used

| Module                  | Algorithm               | Library       | File                         |
| ----------------------- | ----------------------- | ------------- | ---------------------------- |
| Symptom Prediction      | Support Vector Classifier (SVC) | Scikit-learn | `symptoms_model.pkl`        |
| Male Diabetes           | Logistic Regression     | Scikit-learn  | `male_diabetes_model.pkl`    |
| Female Diabetes         | ML Classifier           | Scikit-learn  | `female_diabetes_model.pkl`  |
| Kidney Stone Detection  | CNN (Deep Learning)     | TensorFlow    | `kidney_model.keras`         |

---

## 📊 Datasets

| File                   | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `Training.csv`         | Symptom-disease training data (132 symptoms, 41 diseases) |
| `Symptom-severity.csv` | Severity scores for each symptom                |
| `symtoms_df.csv`       | Symptom-disease mapping reference               |
| `description.csv`      | Detailed description for each disease           |
| `precautions_df.csv`   | 4 precautionary measures per disease            |
| `medications.csv`      | Recommended medications per disease             |
| `diets.csv`            | Dietary recommendations per disease             |
| `workout_df.csv`       | Workout/exercise recommendations per disease    |

---

## 🔮 Future Enhancements

- ❤️ **Heart Disease Prediction** — model files already present (`heart_rf_model.pkl`, `heart_scaler.pkl`, `heart_features.pkl`)
- 📊 **Dashboard** — analytics and visualization of prediction history
- 🔐 **User Authentication** — login/signup for personalized health tracking
- 📱 **Mobile Responsive** — progressive web app support
- 🧪 **Model Retraining** — interface to upload new data and retrain models
- 📄 **PDF Report Generation** — downloadable health report after prediction

---

## 📝 License

This project is for educational and research purposes only. The predictions made by this system are **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for medical concerns.

---

## 👨‍💻 Author

Developed as part of a Multi-Disease Prediction System project.

---

> ⚠️ **Disclaimer:** This application is intended for educational purposes only. Do not use the predictions for self-diagnosis or treatment decisions. Always seek professional medical advice.
