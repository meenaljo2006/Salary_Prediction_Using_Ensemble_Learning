# 🎒 Salary Predictor

This is a machine learning project that predicts employee salaries based on job-related inputs using ensemble learning techniques. The final model is deployed with a user-friendly Gradio GUI.

---

## 📌 Project Details

- **Project Title:** Salary Prediction using Ensemble Learning  
- **Tech Stack:** Python, Jupyter Notebook, Pandas, Scikit-learn, XGBoost, Gradio  
- **Model Used:** XGBoost Regressor (best performer)  
- **Interface:** Interactive web app built using Gradio  

---

## 📊 Dataset Info

- **Source:** Provided by instructor.  
- **Key Features Used:**
  - Age  
  - Gender  
  - Education Level  
  - Job Title  
  - Experience Level  

---

## 🧠 Machine Learning Pipeline

1. Data Cleaning and Preprocessing  
2. Label Encoding of Categorical Features  
3. Feature Scaling using StandardScaler  
4. Model Training with XGBoost Regressor  
5. Evaluation using R² Score, MAE, and RMSE  
6. GUI Development using Gradio  

---

## 💻 How to Run the App

To run this app locally, install the required libraries and execute the Python GUI script:
```bash
pip install gradio pandas scikit-learn xgboost
python New_Salary_gui.py

```
---

## 📷 Screenshot

<img width="1908" height="1002" alt="Image" src="https://github.com/user-attachments/assets/e4e2d808-a2d0-4e0f-92c4-d033f68b6b2e" />

---

## ✅ Output Sample

User selects age, gender, education level, job title, and experience level  
The app predicts and displays the estimated monthly and yearly salary

---

## 📁 Files in this Repository

- `Employe_Salary_Prediction_using_ML_Project_final.ipynb` – Complete notebook with model building and evaluation  
- `New_Salary_gui.py` – Python file for running the Gradio app  
- `xgb_model.pkl` – Trained XGBoost model  
- `scaler.pkl` – StandardScaler object used for feature scaling  
- `README.md` – Project documentation (this file)
- `Salary_Data_(salary-prediction-project)`- Dataset
- `salary_prediction_gui_screenshot`- Output Screenshort

---

## 📝 Conclusion

This project demonstrates how ensemble learning, specifically using XGBoost, can effectively predict salaries based on multiple categorical and numerical inputs. The user-friendly Gradio interface makes it easy to test predictions interactively.
