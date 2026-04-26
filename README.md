# 🎓 Nigerian Student Dropout Predictor

A complete multi-class ML project predicting student academic trajectory across 4 categories — **Graduated, At-Risk, Suspended and Dropped-Out** — trained on 50,100 student records.

## 🌐 Live Demo
**[Try the app →](https://web-dropoutpredictor.up.railway.app)**

---

## 📌 The Honest Story
Every model trained on this dataset — including a Neural Network — plateaued at **~63% accuracy**. This is not a failure. This is one of the most important ML lessons:

> *"No algorithm can extract patterns that don't exist in the data. Garbage In → Garbage Out."*

The training data had excessive noise in class generation, causing all 4 student categories to overlap heavily in feature space. Diagnosing a **data problem vs a model problem** is a senior engineering skill — learned here firsthand.

**An honest 63% with a clear explanation beats a fake 95% any day.**

---

## 📊 Dataset
| Property | Value |
|---|---|
| Rows | 50,100 (largest dataset in portfolio) |
| Columns | 26 |
| Target | StudentStatus: 4 classes |
| Challenge | Multi-class imbalance + noisy data |

### Target Classes
| Class | Meaning |
|---|---|
| Graduated | Student on track to complete degree |
| At-Risk | Warning signs — early intervention needed |
| Suspended | Serious academic jeopardy |
| Dropped-Out | High probability of leaving entirely |

---

## 🧹 Data Cleaning
| Column | Problem | Solution |
|---|---|---|
| CGPA | "3.5/5.0", "3.5 points", outliers ×10 | Strip /5.0, pd.to_numeric, IQR clip |
| AttendanceRate | "75%", "75 percent", negatives | Strip %, clip 0-100 |
| FamilyIncome | "NGN85,000", "₦85,000", outliers ×100 | Strip NGN/₦, commas, IQR clip |
| MentalHealthScore | "7/10", "7 out of 10" | Strip /10, strip suffix |
| DistanceFromCampus | "5km", "3.2miles" | Miles × 1.60934 → km |
| Level | 100/200L/Year 1 — 15 formats | Extract numeric, map Year→×100 |
| StudentStatus | **26 different formats!** | str.capitalize() + dictionary map |

---

## 🤖 All 6 Models Trained
| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 63.68% | Baseline |
| Decision Tree | 61.02% | max_depth=7 |
| Random Forest | 62.15% | 200 trees |
| **XGBoost** | **63.70%** | **Best — deployed** |
| LightGBM | 63.50% | Close second |
| Neural Network | 61.20% | Underfitting confirmed |

All models hit the same ~63% ceiling — confirming the data quality issue, not a pipeline error.

---

## 🧠 Neural Network Architecture
```python
model = keras.Sequential([
    Dense(256, activation='relu', input_shape=(n,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')     # 4 classes
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## ⚙️ ColumnTransformer Pipeline
```python
preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first',
             handle_unknown='ignore'), cat_cols),
    ('scaler', StandardScaler(), num_cols),
], remainder='passthrough')
```
First project using full preprocessing pipeline — saved with joblib for Flask deployment.

---

## 🏗️ Tech Stack
- **Language:** Python
- **ML:** Scikit-learn, XGBoost, LightGBM
- **Deep Learning:** TensorFlow/Keras
- **Web Backend:** Flask
- **Frontend:** HTML5, CSS3
- **Deployment:** Railway.app
- **Version Control:** GitHub

---

## 📁 Project Structure
```
StudentDropoutPredictor/
├── data/
│   └── nigerian_student_dropout_messy.csv
├── models/
│   ├── XGB.joblib
│   └── Preprocessor.joblib
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── APP.py
├── requirements.txt
└── Procfile
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/DavidGabriel213/StudentDropoutPredictor
cd StudentDropoutPredictor
pip install -r requirements.txt
python APP.py
```

---

## 💡 Key Learnings
1. **Data quality > model complexity** — no algorithm fixes fundamentally noisy labels
2. **Underfitting signature** — train and val both plateau = data problem, not model
3. **Training-serving skew** — preprocessor must match between training and Flask
4. **ColumnTransformer** — professional preprocessing pipeline for production
5. **compute_class_weight for Keras** — must convert array to dictionary

---

## 👨‍💻 About
**Gabriel David** | Mathematics Undergraduate | ATBU Bauchi
Self-taught ML Engineer — built during Industrial Training placement.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--david--ds-blue)](https://linkedin.com/in/gabriel-david-ds)
[![GitHub](https://img.shields.io/badge/GitHub-DavidGabriel213-black)](https://github.com/DavidGabriel213)
