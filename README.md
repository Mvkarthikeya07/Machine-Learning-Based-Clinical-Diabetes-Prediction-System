🧠 Clinical Diabetes Prediction System

A full-stack Machine Learning + Flask web application that predicts diabetes risk using real medical diagnostic features.
This project demonstrates a complete ML workflow — from data preprocessing to model training, saving, API integration, and deployment-ready web UI.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python">
  <img src="https://img.shields.io/badge/Flask-Framework-black?logo=flask">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen">
</p>

🚀 Overview

This project demonstrates an end-to-end ML pipeline, including:

🧹 Data preprocessing

🤖 Logistic Regression model

💾 Model saving using Joblib

🔌 Flask API endpoints

🎨 Interactive HTML + CSS UI

📊 Real-time predictions with probability output

A perfect demonstration of AI in healthcare, ML deployment, and Python full-stack development.

🧰 Tech Stack
Backend & Machine Learning

Python

Pandas, NumPy

Scikit-Learn

Joblib

Flask

Frontend

HTML5

CSS3

JavaScript

📂 Project Structure
diabetes_project/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── venv/                  
├── app.py                 # Flask API + Frontend
├── train_model.py         # model training script
├── diabetes_user.csv      # dataset
├── model.joblib           # trained ML model
└── requirements.txt

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction

2. Install dependencies
pip install -r requirements.txt

3. Train the model
python train_model.py

4. Run the Flask app
python app.py

5. Open in browser
http://127.0.0.1:5000/

🧪 Example Predictions
🔴 High-Risk Sample

Input:

Glucose: 180

BMI: 36

Age: 45

Insulin: 94

Pregnancies: 2

Output:

Probability: 77.7%

Prediction: Diabetes Positive (High Risk)

🟢 Low-Risk Sample

Input:

Glucose: 95

BMI: 24

Age: 25

Output:

Probability: 12.4%

Prediction: Diabetes Negative (Low Risk)

📡 API Endpoints
Endpoint	Method	Description
/	GET	Loads frontend UI
/predict	POST	Predicts diabetes + probability
/health	GET	Checks model status
/data-head	GET	Returns first 5 rows of dataset
/data-tail	GET	Returns last 5 rows
🧠 Dataset Description

The dataset diabetes_user.csv contains:

Feature	Description
Pregnancies	Number of pregnancies
Glucose	Plasma glucose concentration
BloodPressure	Diastolic BP
SkinThickness	Skin fold thickness
Insulin	2-hour serum insulin
BMI	Body mass index
DiabetesPedigreeFunction	Hereditary risk score
Age	Age in years
Outcome	1 = Diabetic, 0 = Non-Diabetic
⭐ Features

✔ Clean & responsive UI
✔ Real-time ML predictions
✔ Probability-based output
✔ Preprocessing pipeline (Imputer + Scaler + Model)
✔ Dataset preview endpoints
✔ Modular and extendable code

🔮 Future Enhancements

Deploy on Render / Heroku

Add user authentication

Save user prediction history

SHAP explainability visuals

Data distribution plots (Matplotlib/Seaborn)

Upgrade UI using Bootstrap / Material UI

🖼️ Screenshots

<img width="1366" height="768" alt="2025-11-26 (2)" src="https://github.com/user-attachments/assets/f29980cb-f5ee-4b6e-87f5-570ba6106c38" />

<img width="1366" height="768" alt="2025-11-26 (1)" src="https://github.com/user-attachments/assets/6754634a-b823-4f18-b541-f2dc565c67f5" />

<img width="1366" height="768" alt="2025-11-26" src="https://github.com/user-attachments/assets/c723845d-7638-47de-9b93-0f43fd7e453e" />


👨‍💻 Author

M V Karthikeya
Machine Learning & Python Developer

📜 License

This project is licensed under the MIT License.

⭐ Support

If you found this useful, please star the repository — it motivates me to build more! ⭐
