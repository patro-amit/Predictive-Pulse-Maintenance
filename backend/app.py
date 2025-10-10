"""
FastAPI Backend for Predictive Maintenance
Topic: "Predictive Maintenance Strategies Using Big Data And Machine Learning"
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE), "frontend")

# Load all available models      // .\run_app.bat
LOADED_MODELS = {}
FEATURES = None

def load_all_models():
    """Load all trained models from the models directory."""
    global FEATURES, LOADED_MODELS
    
    model_files = ["model.pkl", "random_forest.pkl", "xgboost.pkl", 
                   "catboost.pkl", "gradient_boosting.pkl"]
    
    print("\n" + "="*70)
    print("🔄 LOADING MODELS...")
    print("="*70)
    
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(model_path):
            try:
                print(f"\n📂 Loading {model_file}...")
                bundle = joblib.load(model_path)
                model_name = model_file.replace(".pkl", "")
                LOADED_MODELS[model_name] = bundle
                
                # Use features from the first loaded model
                if FEATURES is None:
                    FEATURES = bundle.get("features", [])
                
                # Show model metrics
                metrics = bundle.get("metrics", {})
                accuracy = metrics.get("accuracy", 0)
                print(f"✅ SUCCESS: {model_name}")
                print(f"   Accuracy: {accuracy*100:.2f}%")
                
            except Exception as e:
                print(f"❌ FAILED to load {model_file}")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {str(e)}")
                import traceback
                print(f"   Full traceback:")
                traceback.print_exc()
        else:
            print(f"⚠️  File not found: {model_file}")
    
    # Try loading feature list if FEATURES is still None
    if FEATURES is None:
        feats_path = os.path.join(MODELS_DIR, "feature_list.json")
        if os.path.exists(feats_path):
            with open(feats_path) as f:
                FEATURES = json.load(f)
    
    print("\n" + "="*70)
    print(f"✅ LOADED {len(LOADED_MODELS)} MODELS: {list(LOADED_MODELS.keys())}")
    print(f"📊 Features count: {len(FEATURES) if FEATURES else 0}")
    print("="*70 + "\n")

# Load models on startup
load_all_models()

# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    version="3.0.0",
    description="Big Data & Machine Learning for Predictive Maintenance"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files for frontend
try:
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")
except:
    pass  # Directory might not exist yet

class PredictRequest(BaseModel):
    inputs: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # Fix Pydantic warning
    
    status: str
    api: str
    version: str
    models_loaded: int
    model_names: List[str]
    features_count: int
    project: str

class ModelInfo(BaseModel):
    name: str
    metrics: Optional[Dict[str, Any]] = None
    features: List[str]

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the main HTML UI."""
    html_path = os.path.join(FRONTEND_DIR, "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return f.read()
    return """
    <html>
        <head><title>Predictive Maintenance</title></head>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>🏭 Predictive Maintenance System</h1>
            <p>Big Data & Machine Learning Strategies</p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Check API health and loaded models status."""
    return {
        "status": "ok",
        "api": "predictive-maintenance",
        "version": "3.0.0",
        "models_loaded": len(LOADED_MODELS),
        "model_names": list(LOADED_MODELS.keys()),
        "features_count": len(FEATURES) if FEATURES else 0,
        "project": "Predictive Maintenance Strategies Using Big Data And Machine Learning"
    }

@app.get("/schema", tags=["General"])
def schema():
    """Get feature schema for the models."""
    if not FEATURES:
        raise HTTPException(status_code=500, detail="Features not loaded")
    
    return {
        "features": [{"name": c, "dtype": "float"} for c in FEATURES],
        "total_features": len(FEATURES)
    }

@app.get("/models", tags=["Models"])
def models():
    """List all available models with their metrics, sorted by accuracy."""
    if not LOADED_MODELS:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    models_info = []
    for name, bundle in LOADED_MODELS.items():
        metrics = bundle.get("metrics", {})
        models_info.append({
            "name": name,
            "accuracy": metrics.get("accuracy", 0),
            "f1_score": metrics.get("f1", 0),
            "auroc": metrics.get("auroc", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "features_count": len(bundle.get("features", []))
        })
    
    # Sort by accuracy descending
    models_info.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
    
    return {
        "models": models_info,
        "best_model": models_info[0]["name"] if models_info else None,
        "status": "ready"
    }

def prepare_input(inputs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Prepare input data for prediction."""
    if not FEATURES:
        raise HTTPException(status_code=500, detail="Features not loaded")
    
    df = pd.DataFrame(inputs)
    
    # Ensure all features are present
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    
    # Reorder columns and convert to numeric
    X = df.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    
    # Fill NaN with 0
    X = X.fillna(0)
    
    return X

@app.post("/predict/compare", tags=["Prediction"])
def predict_compare(body: PredictRequest):
    """Compare predictions from all available models."""
    if not LOADED_MODELS:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    try:
        X = prepare_input(body.inputs)
        
        comparison = []
        for model_name, bundle in LOADED_MODELS.items():
            model = bundle["pipeline"]
            proba = model.predict_proba(X)[:, 1]
            labels = (proba >= 0.5).astype(int)
            
            comparison.append({
                "model": model_name,
                "predictions": ["Needs_Maintenance" if i else "Working" for i in labels],
                "probability": [float(p) for p in proba],  # Fixed: changed from positive_proba
                "confidence": [float(max(p, 1-p)) for p in proba],
                "metrics": bundle.get("metrics", {})
            })
        
        return {
            "comparison": comparison,
            "num_models": len(comparison)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")

@app.post("/predict", tags=["Prediction"])
def predict(body: PredictRequest):
    """Make predictions using the best model (highest accuracy)."""
    if not LOADED_MODELS:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    # Find best model based on accuracy
    best_model_name = None
    best_accuracy = 0
    for name, bundle in LOADED_MODELS.items():
        metrics = bundle.get("metrics", {})
        accuracy = metrics.get("accuracy", 0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
    
    # Fallback to first model if no metrics found
    if best_model_name is None:
        best_model_name = list(LOADED_MODELS.keys())[0]
    
    bundle = LOADED_MODELS[best_model_name]
    model = bundle["pipeline"]
    
    try:
        X = prepare_input(body.inputs)
        
        # Get predictions
        proba = model.predict_proba(X)[:, 1]
        labels = (proba >= 0.5).astype(int)
        
        return {
            "model_used": best_model_name,
            "predictions": ["Needs_Maintenance" if i else "Working" for i in labels],
            "probability": [float(p) for p in proba],  # Fixed: changed from positive_proba
            "confidence": [float(max(p, 1-p)) for p in proba],
            "model_metrics": bundle.get("metrics", {})
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/{model_name}", tags=["Prediction"])
def predict_with_model(model_name: str, body: PredictRequest):
    """Make predictions using a specific model."""
    if model_name not in LOADED_MODELS:
        available = list(LOADED_MODELS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {available}"
        )
    
    bundle = LOADED_MODELS[model_name]
    model = bundle["pipeline"]
    
    try:
        X = prepare_input(body.inputs)
        
        # Get predictions
        proba = model.predict_proba(X)[:, 1]
        labels = (proba >= 0.5).astype(int)
        
        return {
            "model_used": model_name,
            "predictions": ["Needs_Maintenance" if i else "Working" for i in labels],
            "probability": [float(p) for p in proba],  # Fixed: changed from positive_proba
            "confidence": [float(max(p, 1-p)) for p in proba],
            "model_metrics": bundle.get("metrics", {})
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/example", tags=["General"])
def example():
    """Get an example with realistic sensor values."""
    if not FEATURES:
        return {"error": "Features not loaded"}
    
    # Create example with realistic sensor values (not zeros)
    example = {
        "temperature_1": 22.5,
        "temperature_2": 23.1,
        "pressure_1": 101.3,
        "pressure_2": 102.7,
        "vibration_1": 0.05,
        "vibration_2": 0.06,
        "rpm_1": 1500.0,
        "rpm_2": 1520.0,
        "cycle": 150.0,
        "operating_hours": 3500.0
    }
    
    # Fill remaining features with realistic values
    for feat in FEATURES:
        if feat not in example:
            if "temp" in feat.lower():
                example[feat] = 22.0
            elif "pressure" in feat.lower():
                example[feat] = 101.0
            elif "vibration" in feat.lower():
                example[feat] = 0.05
            elif "rpm" in feat.lower():
                example[feat] = 1500.0
            else:
                example[feat] = 50.0
    
    return {
        "inputs": [example],
        "note": "Example with realistic sensor readings"
    }

@app.get("/example-payload", tags=["General"])
def example_payload():
    """Get an example prediction payload structure."""
    if not FEATURES:
        return {"error": "Features not loaded"}
    
    # Create example with realistic values
    example = {feat: 0.0 for feat in FEATURES}
    
    return {
        "inputs": [example],
        "note": "Replace values with your actual sensor readings"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
