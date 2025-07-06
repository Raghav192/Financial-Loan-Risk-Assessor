from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import numpy as np
import joblib
import shap

# --- Global Setup ---
app = FastAPI(title="Predictive Model For Financial Loan Risk Assesment API")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load model components at startup
try:
    model_pipeline = joblib.load("models/best_model.joblib")
    explainer = joblib.load("models/shap_explainer.joblib")
    feature_names = joblib.load("models/processed_feature_names.joblib")
    print("INFO:     ✅ Model components loaded successfully.")
except Exception as e:
    model_pipeline, explainer, feature_names = None, None, None
    print(f"FATAL:    ❌ Could not load model files. Error: {e}")

# Define the expected input structure and validation rules
class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0); annual_inc: float = Field(..., gt=0); dti: float; int_rate: float = Field(..., gt=0)
    emp_length: float; credit_history_length: float
    term: Literal[' 36 months', ' 60 months']; grade: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G']
    home_ownership: Literal['RENT', 'MORTGAGE', 'OWN', 'ANY']; purpose: Literal['debt_consolidation', 'credit_card', 'home_improvement', 'other']
    sub_grade: str = "C1"; verification_status: str = "Verified"; open_acc: float = 10.0; pub_rec: float = 0.0; revol_bal: float = 15000.0
    revol_util: float = 50.0; total_acc: float = 25.0; initial_list_status: str = "w"; application_type: str = "Individual"; mort_acc: float = 2.0
    pub_rec_bankruptcies: float = 0.0; installment: float = 332.14

FEATURE_ORDER = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length',
    'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'dti', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
    'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'credit_history_length',
    # New features from A++ notebook
    'loan_to_income_ratio', 'interest_to_income_ratio', 'revol_util_to_open_acc'
]

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def api_predict(application: LoanApplication):
    if model_pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded. Check server logs."})

    # --- Feature Engineering for new features ---
    input_dict = application.model_dump()
    input_dict['loan_to_income_ratio'] = input_dict['loan_amnt'] / (input_dict['annual_inc'] + 1)
    input_dict['interest_to_income_ratio'] = (input_dict['installment'] * 12) / (input_dict['annual_inc'] + 1)
    input_dict['revol_util_to_open_acc'] = input_dict['revol_util'] / (input_dict['open_acc'] + 1)
    
    input_df = pd.DataFrame([input_dict], columns=FEATURE_ORDER)
    
    risk_proba = model_pipeline.predict_proba(input_df)[0][1]
    
    explanation_data = []
    try:
        processed_input = model_pipeline.named_steps['preprocessor'].transform(input_df)
        shap_values = explainer.shap_values(processed_input)
        
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values[0],
            'feature_value': processed_input[0]
        })
        shap_df_filtered = shap_df[shap_df['feature_value'] != 0].copy()
        shap_df_filtered['abs_shap'] = np.abs(shap_df_filtered['shap_value'])
        top_contributors = shap_df_filtered.sort_values(by='abs_shap', ascending=False).head(3)

        for _, row in top_contributors.iterrows():
            # THE FIX: Aggressively convert every potential numpy float to a standard Python float
            shap_value = float(row['shap_value']) 
            explanation_data.append({
                "feature": row['feature'].split('__')[1],
                "impact": "increases" if shap_value > 0 else "decreases"
            })
    except Exception as e:
        print(f"--- SHAP Explanation Generation Error: {e} ---")
        explanation_data.append({"feature": "Explanation currently unavailable", "impact": "neutral"})
    
    # THE FIX: Aggressively convert the final risk probability to a standard Python float
    risk_score = float(risk_proba) * 100
    
    if risk_score > 50: recommendation, risk_class = "High Risk - Not Recommended", "high-risk"
    elif risk_score > 20: recommendation, risk_class = "Medium Risk - Manual Review", "medium-risk"
    else: recommendation, risk_class = "Low Risk - Recommended", "low-risk"

    return JSONResponse(content={
        "risk_score": risk_score,
        "recommendation": recommendation,
        "risk_class": risk_class,
        "explanation": explanation_data,
    })