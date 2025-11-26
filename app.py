"""
Incident Management SLA Breach Prediction - FastAPI Application
RESTful API for real-time SLA breach predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Initialize FastAPI app
app = FastAPI(
    title="Incident SLA Breach Prediction API",
    description="Predicts SLA breach probability for incident management using ML",
    version="1.0.0"
)

# Load model
MODEL_PATH = 'models/best_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Feature names (must match training)
FEATURES = [
    'reassignment_count', 'reopen_count', 'sys_mod_count',
    'open_hour', 'open_day', 'open_month',
    'is_business_hours', 'is_weekday',
    'workload_score', 'complexity_flag',
    'reassigned', 'reopened',
    'incident_state_code', 'active'
]

# Pydantic models for request/response
class IncidentInput(BaseModel):
    """Input schema for SLA breach prediction"""
    reassignment_count: int = Field(..., ge=0, le=30, description="Number of reassignments")
    reopen_count: int = Field(..., ge=0, le=10, description="Number of times reopened")
    sys_mod_count: int = Field(..., ge=0, le=150, description="System modification count")
    open_hour: int = Field(..., ge=0, le=23, description="Hour incident opened (0-23)")
    open_day: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    open_month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    is_business_hours: int = Field(..., ge=0, le=1, description="During business hours? (0/1)")
    is_weekday: int = Field(..., ge=0, le=1, description="Is weekday? (0/1)")
    workload_score: float = Field(..., ge=0, description="Calculated workload score")
    complexity_flag: int = Field(..., ge=0, le=1, description="High complexity? (0/1)")
    reassigned: int = Field(..., ge=0, le=1, description="Has been reassigned? (0/1)")
    reopened: int = Field(..., ge=0, le=1, description="Has been reopened? (0/1)")
    incident_state_code: int = Field(..., ge=0, description="Encoded incident state")
    active: int = Field(..., ge=0, le=1, description="Is incident active? (0/1)")
    
    class Config:
        schema_extra = {
            "example": {
                "reassignment_count": 2,
                "reopen_count": 0,
                "sys_mod_count": 8,
                "open_hour": 15,
                "open_day": 2,
                "open_month": 11,
                "is_business_hours": 1,
                "is_weekday": 1,
                "workload_score": 2.8,
                "complexity_flag": 0,
                "reassigned": 1,
                "reopened": 0,
                "incident_state_code": 1,
                "active": 1
            }
        }

class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    prediction: str
    sla_breach_probability: float
    sla_met_probability: float
    risk_level: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

# API Endpoints

@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Incident SLA Breach Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            h1 { color: #e74c3c; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            code { background: #34495e; color: #ecf0f1; padding: 2px 6px; border-radius: 3px; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚨 Incident SLA Breach Prediction API</h1>
            <p>Machine Learning API for predicting Service Level Agreement breaches in incident management</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Check API health status and model availability</p>
            </div>
            
            <div class="endpoint">
                <h3>POST /predict</h3>
                <p>Predict SLA breach probability for an incident</p>
                <p><strong>Required fields:</strong> reassignment_count, reopen_count, sys_mod_count, temporal features, workload metrics</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /docs</h3>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /model-info</h3>
                <p>Get model metadata and feature information</p>
            </div>
            
            <p><a href="/docs">📚 Try the API interactively →</a></p>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "description": "Predicts SLA breach probability based on incident characteristics",
        "target": "sla_breach (0=Met, 1=Breach)",
        "feature_info": {
            "reassignment_count": "Number of times incident was reassigned",
            "reopen_count": "Number of times incident was reopened",
            "sys_mod_count": "Total system modifications",
            "workload_score": "Composite workload metric",
            "complexity_flag": "High complexity indicator (0/1)",
            "incident_state_code": "Encoded incident state",
            "active": "Is incident currently active (0/1)"
        }
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: IncidentInput):
    """
    Predict SLA breach probability for an incident
    
    Returns:
    - prediction: 'SLA_Breach' or 'SLA_Met'
    - sla_breach_probability: Probability of SLA breach (0-1)
    - sla_met_probability: Probability of meeting SLA (0-1)
    - risk_level: 'low', 'medium', 'high', or 'critical'
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        features_df = pd.DataFrame([input_dict])
        
        # Ensure correct feature order
        features_df = features_df[FEATURES]
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # Extract probabilities
        sla_met_prob = float(probabilities[0])
        sla_breach_prob = float(probabilities[1])
        
        # Determine risk level
        if sla_breach_prob >= 0.8:
            risk_level = "critical"
        elif sla_breach_prob >= 0.6:
            risk_level = "high"
        elif sla_breach_prob >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Prepare response
        result = {
            "prediction": "SLA_Breach" if prediction == 1 else "SLA_Met",
            "sla_breach_probability": round(sla_breach_prob, 4),
            "sla_met_probability": round(sla_met_prob, 4),
            "risk_level": risk_level,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
def batch_predict(data: list[IncidentInput]):
    """
    Batch prediction endpoint for multiple incidents
    Maximum 100 incidents per batch
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")
    
    try:
        results = []
        
        for item in data:
            input_dict = item.dict()
            features_df = pd.DataFrame([input_dict])
            features_df = features_df[FEATURES]
            
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            
            # Risk level
            breach_prob = float(probabilities[1])
            if breach_prob >= 0.8:
                risk = "critical"
            elif breach_prob >= 0.6:
                risk = "high"
            elif breach_prob >= 0.4:
                risk = "medium"
            else:
                risk = "low"
            
            results.append({
                "prediction": "SLA_Breach" if prediction == 1 else "SLA_Met",
                "sla_breach_probability": round(breach_prob, 4),
                "risk_level": risk
            })
        
        return {
            "count": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    print("="*70)
    print("INCIDENT SLA BREACH PREDICTION API STARTING")
    print("="*70)
    print(f"Model loaded: {model is not None}")
    print(f"Features: {FEATURES}")
    print("API ready at http://localhost:8000")
    print("Docs available at http://localhost:8000/docs")
    print("="*70)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )