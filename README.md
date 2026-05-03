# 🏨 Hotel Booking Cancellation Prediction — ANN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-ANN-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Deployment-Streamlit%20Web%20App-red?style=for-the-badge"/>
</p>

---

## 📌 Project Overview

This project builds an **Artificial Neural Network (ANN)** to predict whether a hotel booking will be **cancelled or confirmed**, based on booking details such as lead time, room type, meal plan, and market segment.

The trained model is deployed as an interactive **Streamlit web application** where users can enter booking details and get an instant prediction along with a downloadable report.

---

## 🎯 Problem Statement

Hotel cancellations are a major challenge for the hospitality industry, leading to revenue loss and poor resource planning. The goal is to build a deep learning model that can:

- **Predict** if a hotel booking will be cancelled before arrival
- **Help hotels** take proactive actions like overbooking strategies or targeted offers
- **Provide a user-friendly interface** for real-time predictions

---

## 📂 Project Structure

```
Hotel-Booking-Cancellation-ANN/
│
├── Dataset/                          # Raw dataset
├── Hotel_Booking_status_ANN.ipynb    # Model training notebook
├── app.py                            # Streamlit web application
├── hotel_ann_model.keras             # Saved trained ANN model
├── scaler.pkl                        # Saved StandardScaler
├── requirements.txt                  # Project dependencies
└── README.md
```

---

## 🗃️ Dataset

| Feature | Description |
|---|---|
| `no_of_adults` | Number of adults in the booking |
| `no_of_children` | Number of children |
| `no_of_weekend_nights` | Weekend nights booked |
| `no_of_week_nights` | Weekday nights booked |
| `type_of_meal_plan` | Meal plan selected |
| `room_type_reserved` | Type of room reserved |
| `lead_time` | Days between booking and arrival |
| `arrival_month` | Month of arrival |
| `market_segment_type` | Booking channel (Online, Offline, Corporate, etc.) |
| `avg_price_per_room` | Average price per room per night |
| `booking_status` | **Target** — Cancelled / Not Cancelled |

---

## 🧠 Model Architecture — Artificial Neural Network (ANN)

```
Input Layer  →  Dense(64, ReLU)
             →  Dropout(0.3)
             →  Dense(32, ReLU)
             →  Dropout(0.2)
             →  Dense(16, ReLU)
Output Layer →  Dense(1, Sigmoid)
```

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Activation (Output):** Sigmoid (for binary classification)  
- **Scaling:** StandardScaler (saved as `scaler.pkl`)

---

## ⚙️ Workflow

```
Raw Data
   │
   ▼
EDA & Data Cleaning
   │
   ▼
Feature Engineering & Label Encoding
   │
   ▼
Train-Test Split + Feature Scaling
   │
   ▼
ANN Model Training
   │
   ▼
Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
   │
   ▼
Save Model (.keras) + Scaler (.pkl)
   │
   ▼
Streamlit Web App Deployment
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Training Accuracy | ~85%+ |
| Validation Accuracy | ~84%+ |
| Loss Function | Binary Crossentropy |
| Evaluation Metrics | Accuracy, Confusion Matrix, Classification Report |

> 📝 Exact metrics are available in the notebook: `Hotel_Booking_status_ANN.ipynb`

---

## 🖥️ Streamlit Web Application

The `app.py` file deploys a fully interactive prediction interface with:

- 📋 **Input form** for all booking features (adults, children, lead time, room type, meal plan, market segment, price)
- 🔍 **Instant prediction** — Cancelled ❌ or Confirmed ✅
- 📄 **Prediction report** preview with all input details
- 📥 **Download button** to save the report as a `.txt` file

### Run the App Locally

```bash
# 1. Clone the repository
git clone https://github.com/pavan-ahire/Hotel-Booking-Cancellation-ANN.git
cd Hotel-Booking-Cancellation-ANN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | ANN model building & training |
| NumPy & Pandas | Data manipulation |
| Scikit-learn | Preprocessing, evaluation metrics |
| Matplotlib & Seaborn | Data visualization & EDA |
| Joblib | Saving/loading the scaler |
| Streamlit | Web app deployment |

---

## 📦 Requirements

```
tensorflow
keras
streamlit
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
```

> Install all dependencies: `pip install -r requirements.txt`

---

## 🔗 Related Projects

| Project | Architecture | Description |
|---|---|---|
| 🧠 [Hotel Booking Cancellation — ANN](https://github.com/pavan-ahire/Hotel-Booking-Cancellation-ANN) | **ANN** | Tabular data binary classification *(this project)* |
| 📝 *(RNN Project)* | **RNN / LSTM** | Sequential / time-series data |
| 🖼️ *(CNN Project)* | **CNN** | Image classification / computer vision |

---

## 👤 Author

**Pavan Suresh Ahire**  
Aspiring Data Scientist | Python • SQL • Power BI • ML • DL • NLP

[![GitHub](https://img.shields.io/badge/GitHub-pavan--ahire-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/pavan-ahire)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-pavan--ahire-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pavan-ahire-260940364/)

---

## ⭐ Show Some Love

If you found this project helpful, please consider giving it a **⭐ Star** on GitHub — it motivates me to keep building!

---

*📌 Part of my Deep Learning project series — ANN | RNN | CNN*
