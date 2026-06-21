# Heart Failure Risk Prediction Using Deep Learning

## Description

This project presents an end-to-end Deep Learning workflow for predicting heart failure risk using structured clinical data.

The model is implemented using TensorFlow/Keras and performs binary classification to determine whether a patient is likely to develop heart disease based on medical attributes such as age, blood pressure, ECG results, and exercise-related indicators.

The project demonstrates key Machine Learning and Deep Learning concepts including:

- Data preprocessing
- Feature engineering
- One-hot encoding
- Feature scaling
- Neural network architecture design
- Model training and evaluation
- Model persistence and inference

---

## Technologies

```text
Python
TensorFlow
Keras
Pandas
NumPy
Scikit-Learn
```

---

## Machine Learning Workflow

```text
Clinical Dataset
        |
        v
Data Preprocessing
        |
        v
One-Hot Encoding
        |
        v
Feature Scaling
        |
        v
Neural Network Training
        |
        v
Model Serialization (.h5)
        |
        v
Inference
        |
        v
Performance Evaluation
```

---

## Dataset Features

The model uses patient health indicators including:

```text
Age
Sex
ChestPainType
RestingBP
Cholesterol
FastingBS
RestingECG
MaxHR
ExerciseAngina
Oldpeak
ST_Slope
```

Target variable:

```text
HeartDisease
```

---

## Model Architecture

The neural network consists of:

```python
Input Layer
      ↓
Dense(6, activation="relu")
      ↓
Dense(6, activation="relu")
      ↓
Dense(1, activation="sigmoid")
```

Output interpretation:

```text
0 = No Heart Disease
1 = Heart Disease
```

---

## Project Structure

```text
deep-learning-heart-failure-prediction/

├── data/
│   └── heart.csv
│
├── src/
│   └── train_and_evaluate.py
│
├── model.h5
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/karolinasniezek/deep-learning-heart-failure-prediction.git
cd deep-learning-heart-failure-prediction
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Training

To train the model from scratch:

```python
model.fit(
    X_train,
    y_train,
    validation_split=0.33,
    batch_size=10,
    epochs=100
)
```

Save trained model:

```python
model.save("model.h5")
```

---

## Model Persistence

The trained model is stored as:

```text
model.h5
```

Loading a trained model:

```python
model = tf.keras.models.load_model("model.h5")
```

This allows the model to be reused for inference and evaluation without retraining.

---

## Inference

Generate prediction probabilities:

```python
y_pred = model.predict(X_test)
```

Convert probabilities to binary predictions:

```python
y_pred = y_pred > 0.5
```

Interpretation:

```text
Probability > 0.5  -> Heart Disease
Probability ≤ 0.5  -> No Heart Disease
```

---

## Model Evaluation

Classification performance is evaluated using Accuracy Score.

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)
```

Example output:

```text
Model Accuracy: 0.84
```

---

## Key Skills Demonstrated

```text
Deep Learning
Machine Learning
Binary Classification
Predictive Analytics
Healthcare Analytics
Feature Engineering
Data Preprocessing
TensorFlow
Keras
Model Persistence
Model Evaluation
```

---

## Future Improvements

Potential extensions for this project:

```text
ROC-AUC Evaluation
Confusion Matrix
Precision / Recall / F1 Score
Hyperparameter Tuning
Explainable AI (SHAP)
Streamlit Frontend
FastAPI Deployment
Docker Containerization
```

---

Machine Learning / Deep Learning Portfolio Project

This project demonstrates the complete machine learning workflow from data preprocessing and neural network training to model persistence, inference, and performance evaluation.
