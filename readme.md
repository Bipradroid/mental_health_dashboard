# 💙 Mental Health Analytics & AI Support Platform

An end-to-end Mental Health Analytics Platform built using Machine Learning, Data Analytics, Generative AI, and Interactive Visualization.

The platform helps users assess their mental well-being, understand stress and isolation patterns, receive personalized insights, interact with an AI mental health assistant, and contribute anonymized data to improve future predictions.

---

# 🚀 Features

## 📊 Interactive Mental Health Dashboard

- Mental health trend analysis
- Country-wise insights
- Occupation-wise analysis
- Gender-based comparisons
- Stress and Isolation monitoring
- Interactive visualizations using Plotly

---

## 🧠 Mental Health Assessment

Users can complete a guided assessment form containing:

- Gender
- Country
- Occupation
- Family History
- Treatment History
- Mood Swings
- Increasing Stress
- Social Weakness
- Work Interest
- Mental Health History
- Coping Struggles
- Days Indoors
- Care Options
- Mental Health Interview

---

## ⚙️ Feature Engineering

Custom features are generated from user responses:

### Stress Score

```python
StressScore = (
    0.5 * IncreasingStress
    + CopingStruggles
    + 0.5 * MoodSwings
    + 0.5 * HabitsChange
    + 0.5 * SocialWeakness
)
```

### Isolation Environment Score

```python
Isolation_env = (
    0.25 * DaysIndoors
    + 0.5 * SocialWeakness
)
```

### Early Stage Indicator

```python
EarlyStage = 1 if Occupation == "Student" else 0
```

---

## 🤖 Machine Learning Prediction Engine

The platform predicts the likelihood of mental health treatment engagement using:

- XGBoost Classifier
- Feature Engineering
- Encoded Demographic Features

### Prediction Outputs

- Treatment Likelihood
- Prediction Confidence
- Stress Score Analysis
- Isolation Score Analysis
- Comparison with Population
- Key Contributing Factors

---

## 📈 Assessment Report

Personalized reports include:

### Stress Analysis

- User Stress Score
- Population Average
- Distribution Comparison

### Isolation Analysis

- User Isolation Score
- Population Average
- Distribution Comparison

### Confidence Analysis

- Model Confidence Gauge
- Treatment Prediction

### Explainability

Top decision-making features such as:

- Family History
- Care Options
- Mental Health Interview
- Self Employment
- Stress Indicators

---

## 🤝 Community Contribution System

Users can voluntarily contribute assessment responses.

Contributed records are:

- Automatically encoded
- Feature engineered
- Stored in a contribution dataset

This allows future retraining and continuous dataset growth.

---

## 💬 AI Mental Health Assistant

Powered by:

- Gemini 2.5 Flash

Capabilities:

- Mental health guidance
- Stress management suggestions
- Personalized recommendations
- Context-aware conversations
- Assessment-aware responses
- Conversation memory

The chatbot automatically switches between:

### General Mode

For users who have not completed an assessment.

### Personalized Mode

Uses:

- Assessment Results
- Stress Score
- Isolation Score
- Prediction Results
- Chat History

to provide personalized support.

---

# 🛠️ Tech Stack

## Frontend

- Streamlit
- HTML
- CSS

## Data Analysis

- Pandas
- NumPy
- Matplotlib

## Visualization

- Plotly

## Machine Learning

- Scikit-Learn
- XGBoost

## Generative AI

- Gemini 2.5 Flash API

## Model Persistence

- Pickle

---

# 📂 Project Structure

```text
mental_health_dashboard/

│
├── app.py
├── chatbot.py
├── mental_health_model(1).pkl
├── feature_columns(1).pkl
├── user_contribution.csv
├── requirements.txt
├── .env
│
└── mental_health_updated1.csv
```

---

# 🔄 Workflow

```text
User
 ↓
Mental Health Assessment
 ↓
Feature Engineering
 ↓
XGBoost Prediction
 ↓
Assessment Report
 ↓
AI Assistant
 ↓
Community Contribution
 ↓
Dataset Growth
 ↓
Future Model Retraining
```

---

# 📌 Future Improvements

- Automated Model Retraining
- PDF Assessment Reports
- Mental Health Knowledge Base (RAG)
- Image Upload Analysis
- Journal/PDF Analysis
- User Authentication
- Progress Tracking
- Personalized Mental Health Plans
- SHAP Explainability
- Multi-language Support

---

# 🎯 Impact

This platform aims to:

- Increase mental health awareness
- Provide accessible self-assessment tools
- Deliver personalized AI-powered guidance
- Encourage early intervention
- Build a continuously improving mental health dataset

---

# 👨‍💻 Author

Bipradeep Biswas

Machine Learning | Data Science | Generative AI
