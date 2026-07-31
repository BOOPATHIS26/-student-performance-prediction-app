# Student Performance Prediction System

A Machine Learning web application that predicts a student's total academic score based on study habits and classroom performance. The application is built using Python, Scikit-learn, and Streamlit, allowing users to receive instant predictions through a simple and interactive interface.

---

## Project Overview

Student academic performance depends on several factors such as study time, attendance, and classroom participation. This project uses a Machine Learning regression model to predict the expected total score based on these factors.

The application provides real-time predictions using a trained model deployed with Streamlit.

---

## Features

- Predicts student's total score instantly
- Interactive Streamlit web application
- Machine Learning regression model
- Feature scaling using Scikit-learn
- Fast and accurate predictions
- Simple and user-friendly interface

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Joblib

---

## Project Structure

```
Student-Performance-Prediction/
│
├── app3.py                  # Streamlit Application
├── student.ipynb            # Model Training Notebook
├── student.csv              # Dataset
├── student_model.pkl        # Trained Machine Learning Model
├── scaler.pkl               # Feature Scaler
├── model_columns.pkl        # Model Columns
├── best_model.pkl           # Best Performing Model
└── README.md
```

---

## Input Features

The application predicts the student's total score using the following inputs:

- Weekly Self Study Hours
- Attendance Percentage
- Class Participation (0–10)

---

## Output

The model predicts:

- Student Total Score

---

## Machine Learning Workflow

1. Collect the student dataset.
2. Perform data cleaning and preprocessing.
3. Select relevant features.
4. Apply feature scaling.
5. Train the regression model.
6. Evaluate model performance.
7. Save the trained model using Joblib.
8. Deploy the model with Streamlit.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/student-performance-prediction.git
```

### Navigate to the Project Folder

```bash
cd student-performance-prediction
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app3.py
```

---

## Application Preview

The application accepts:

- Weekly Self Study Hours
- Attendance Percentage
- Class Participation

It then predicts the student's total academic score in real time.

---

## Future Enhancements

- Include additional academic and personal performance factors
- Improve model accuracy using ensemble learning techniques
- Add data visualization dashboards
- Compare multiple machine learning models
- Deploy the application on Streamlit Cloud
- Generate personalized study recommendations

---

## Skills Demonstrated

- Machine Learning
- Data Preprocessing
- Feature Scaling
- Regression
- Model Deployment
- Streamlit Development
- Python Programming
- Predictive Analytics

---

## Learning Outcomes

This project helped me gain practical experience in:

- Building an end-to-end Machine Learning pipeline
- Training and evaluating regression models
- Deploying Machine Learning models using Streamlit
- Data preprocessing and feature scaling
- Developing interactive web applications for prediction

---

## Author

**Boopathi S**

Aspiring AI/ML Engineer | Java Developer

LinkedIn: https://linkedin.com/in/your-profile

GitHub: https://github.com/yourusername

---

If you found this project useful, consider giving it a star.
