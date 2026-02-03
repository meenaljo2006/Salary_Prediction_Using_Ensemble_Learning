import gradio as gr
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for dropdown values
df = pd.read_csv("Salary_Data_(salary-prediction-project).csv")
gender_options = sorted(df['Gender'].dropna().unique())
education_options = sorted(df['Education Level'].dropna().unique())
job_title_options = sorted(df['Job Title'].dropna().unique())

# Prepare label encoders
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_job = LabelEncoder()
le_gender.fit(df['Gender'])
le_edu.fit(df['Education Level'])
le_job.fit(df['Job Title'])

# Prediction function
def predict_salary(age, gender, edu, job, exp):
    gender_encoded = le_gender.transform([gender])[0]
    edu_encoded = le_edu.transform([edu])[0]
    job_encoded = le_job.transform([job])[0]
    input_df = pd.DataFrame([[age, gender_encoded, edu_encoded, job_encoded, exp]],
                            columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
    scaled_input = scaler.transform(input_df)
    salary = model.predict(scaled_input)[0]
    result_text = f"💰 Predicted Monthly Salary: ₹{salary:,.2f}"
        
    return gr.Textbox(value=result_text, visible=True)

custom_css = """
footer {
    display: none !important;
}
.narrow-layout {
    max-width: 400px !important;
    margin: 0 auto !important;
}
"""

# Interface
with gr.Blocks(css=custom_css) as interface:
    gr.HTML("""
    <h1 style='text-align:center; font-size:32px; margin-bottom:0;'>💼 Salary Predictor</h1>
    <p style='text-align:center; font-size:20px; margin-top:4px;'>Enter your details to get a predicted monthly salary</p>
""")


    with gr.Column(elem_classes=["narrow-layout"]):
        age = gr.Number(label="Age")
        gender = gr.Dropdown(gender_options, label="Gender")
        edu = gr.Dropdown(education_options, label="Education Level")
        job = gr.Dropdown(job_title_options, label="Job Title")
        exp = gr.Number(label="Years of Experience")

    
        submit_btn = gr.Button("Predict")

        output = gr.Textbox(label="Prediction Result",visible=False)

    # Attach function
    submit_btn.click(fn=predict_salary, inputs=[age, gender, edu, job, exp], outputs=output)

    # Button style
    gr.HTML("""
    <style>
        button { 
            background-color: #007bff !important; 
            color: white !important;
        }
    </style>
    """)

interface.launch()
