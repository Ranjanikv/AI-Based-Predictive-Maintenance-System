# AI-Based Predictive Maintenance System

An AI-powered Predictive Maintenance System that uses machine sensor data and an **XGBoost Machine Learning model** to predict machine failures and estimate the probability of failure before a breakdown occurs.

The project also includes a **sensor simulator** for generating different machine conditions and a **Streamlit dashboard** for monitoring machine health and failure probability.

---

## Project Overview

Traditional machine maintenance often depends on fixed maintenance schedules or waiting until a machine fails.

Predictive maintenance takes a different approach.

Instead of waiting for a breakdown, machine sensor readings are analyzed using Machine Learning to identify patterns that may indicate an upcoming failure.

This project uses the **AI4I 2020 Predictive Maintenance Dataset** to train an XGBoost classification model.

The system analyzes parameters such as:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type

and predicts whether a machine is likely to experience a failure.

The model also produces a **failure probability**, which is then used by the application to classify the machine into:

- 🟢 Healthy
- 🟡 At Risk
- 🔴 Critical

---

# Objectives

The main objectives of this project are:

- Predict machine failure using Machine Learning.
- Analyze industrial machine sensor data.
- Build a binary classification model using XGBoost.
- Preprocess and encode categorical machine data.
- Evaluate the model using multiple classification metrics.
- Generate failure probability.
- Simulate changing machine conditions.
- Monitor multiple simulated machines.
- Build an interactive dashboard using Streamlit.
- Categorize machine health based on predicted failure probability.
- Save and reuse the trained Machine Learning model.

---

# Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data loading, manipulation and preprocessing |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Data splitting, encoding and model evaluation |
| **XGBoost** | Machine failure classification |
| **Joblib** | Saving and loading trained models |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive data visualization |
| **Matplotlib** | Visualization and model evaluation |
| **Git** | Version control |
| **GitHub** | Project hosting and collaboration |

---

# Dataset

This project uses the:

**AI4I 2020 Predictive Maintenance Dataset**

The dataset contains machine operating conditions and information about whether a machine failure occurred.

## Features Used

| Feature | Description |
|---|---|
| `Type` | Machine/product type |
| `Air temperature` | Temperature of the surrounding air |
| `Process temperature` | Temperature of the machine process |
| `Rotational speed` | Rotational speed of the machine |
| `Torque` | Torque applied to the machine |
| `Tool wear` | Tool usage/wear |
| `Machine failure` | Target variable |

## Model Evaluation

The trained **XGBoost Classifier** was evaluated using multiple classification metrics instead of relying only on accuracy.

The evaluation metrics used were:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Model Performance

The model achieved an overall accuracy of:

**98.95%**

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| No Failure (0) | 99% | 100% | 99% |
| Machine Failure (1) | 93% | 75% | 83% |

## `File Descriptions`

## app.py

Main Streamlit application.

Responsible for:

- Loading machine data
- Loading the trained XGBoost model
- Generating machine failure predictions
- Calculating failure probability
- Categorizing machine health as Healthy, At Risk, or Critical
- Displaying machine monitoring information
- Creating interactive visualizations
- Presenting the overall predictive maintenance dashboard

---

## train_model.py

Responsible for training the Machine Learning model.

## sensor_simulator.py

Simulates machine sensor readings and changing machine conditions.

It creates simulated machine states such as:

Healthy
Deteriorating
Critical

The simulated sensor values are passed to the trained XGBoost model to generate:

Machine failure predictions
Failure probabilities
Machine health status

This demonstrates how the system could respond to changing machine conditions in a real-time predictive maintenance environment.

## predictive_maintenance_model.pkl

Serialized trained XGBoost classification model.

This file contains the trained Machine Learning model and allows the application to make predictions without retraining the model.

The model predicts:

0 → No Machine Failure

1 → Machine Failure

It also provides class probabilities using predict_proba().

## type_encoder.pkl

Serialized Scikit-learn LabelEncoder.

This encoder converts the categorical Type feature into numerical values that can be processed by the XGBoost model.

The saved encoder ensures that the same encoding used during model training is also used when making predictions on new machine data.

### Workflow

```text
Load Dataset
     ↓
Prepare Features
     ↓
Encode Categorical Data
     ↓
Split Dataset
     ↓
Train XGBoost Model
     ↓
Evaluate Model
     ↓
Save Trained Model
     ↓
Save Type Encoder
