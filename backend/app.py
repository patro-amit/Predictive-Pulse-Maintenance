"""
FastAPI Backend for Predictive Maintenance System
Topic: "Predictive Maintenance Strategies Using Big Data And Machine Learning"

Big Data Tools Integrated:
- Apache Spark (PySpark): Distributed data processing
- MongoDB: NoSQL database for prediction storage and analytics

Research-Grade Features:
- 4 optimized ML models (XGBoost, CatBoost, Random Forest, Gradient Boosting)
- NASA C-MAPSS turbofan dataset (21,974 samples)
- Advanced feature engineering (63 features)
- Varying accuracy levels (suitable for research paper publication)
- Real-time predictions via REST API
- MongoDB for historical data storage
- Apache Spark for big data processing
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
import os
from typing import List, Optional
from datetime import datetime
import time

# Big Data Tool Imports
try:
    from mongodb_config import get_mongodb_handler
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️  MongoDB not available")

try:
    from spark_processor import get_spark_processor
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  Apache Spark not available")
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
    """Load research-grade models (4 models) from the models directory."""
    global FEATURES, LOADED_MODELS
    
    # Research-grade models only: 4 optimized models
    model_files = ["random_forest.pkl", "xgboost.pkl", 
                   "catboost.pkl", "gradient_boosting.pkl"]
    
    print("\n" + "="*70)
    print("🔄 LOADING RESEARCH-GRADE MODELS (NASA C-MAPSS trained)...")
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

# Initialize Big Data tools
mongodb_handler = None
spark_processor = None

if MONGODB_AVAILABLE:
    mongodb_handler = get_mongodb_handler()

if SPARK_AVAILABLE:
    spark_processor = get_spark_processor()

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

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML page"""
    html_path = os.path.join(FRONTEND_DIR, "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    return {"message": "Frontend not found"}

# ============================================================================
# BIG DATA ENDPOINTS - MongoDB & Apache Spark
# ============================================================================

@app.get("/bigdata/status")
async def bigdata_status():
    """Check status of Big Data tools"""
    return {
        "mongodb": {
            "available": MONGODB_AVAILABLE and mongodb_handler is not None,
            "status": "Connected" if (MONGODB_AVAILABLE and mongodb_handler) else "Not Available"
        },
        "apache_spark": {
            "available": SPARK_AVAILABLE and spark_processor is not None,
            "status": "Running" if (SPARK_AVAILABLE and spark_processor and spark_processor.spark) else "Not Available",
            "version": spark_processor.spark.version if (spark_processor and spark_processor.spark) else "N/A"
        }
    }

@app.get("/bigdata/predictions/history")
async def get_prediction_history(limit: int = 100):
    """Get prediction history from MongoDB"""
    if not mongodb_handler:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    
    try:
        predictions = mongodb_handler.get_recent_predictions(limit=limit)
        # Convert ObjectId to string for JSON serialization
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            if 'timestamp' in pred:
                pred['timestamp'] = pred['timestamp'].isoformat()
        return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {str(e)}")

@app.get("/bigdata/predictions/statistics")
async def get_prediction_statistics():
    """Get prediction statistics from MongoDB"""
    if not mongodb_handler:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    
    try:
        stats = mongodb_handler.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {str(e)}")

@app.get("/bigdata/spark/insights")
async def get_spark_insights():
    """Get dataset insights using Apache Spark"""
    if not spark_processor:
        raise HTTPException(status_code=503, detail="Apache Spark not available")
    
    try:
        data_path = os.path.join(os.path.dirname(BASE), "data", "predictive_maintenance_bigdata.csv")
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        insights = spark_processor.get_data_insights(data_path)
        return {
            "tool": "Apache Spark",
            "insights": insights,
            "message": "Data insights computed using distributed processing"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spark error: {str(e)}")

@app.post("/bigdata/spark/process")
async def process_with_spark():
    """Process dataset using Apache Spark"""
    if not spark_processor:
        raise HTTPException(status_code=503, detail="Apache Spark not available")
    
    try:
        data_path = os.path.join(os.path.dirname(BASE), "data", "predictive_maintenance_bigdata.csv")
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        start_time = time.time()
        processed_df = spark_processor.process_large_dataset(data_path)
        processing_time = time.time() - start_time
        
        if processed_df is None:
            raise HTTPException(status_code=500, detail="Spark processing returned None")
        
        return {
            "tool": "Apache Spark",
            "status": "success",
            "rows_processed": len(processed_df),
            "columns": list(processed_df.columns),
            "processing_time_seconds": round(processing_time, 2),
            "message": "Dataset processed using distributed computing with Apache Spark"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spark error: {str(e)}")

# ============================================================================
# END BIG DATA ENDPOINTS
# ============================================================================

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
    """Prepare input data for prediction with feature engineering."""
    if not FEATURES:
        raise HTTPException(status_code=500, detail="Features not loaded")
    
    df = pd.DataFrame(inputs)
    
    # Step 1: Ensure base features exist (fill missing base features with realistic defaults)
    # Use normal operating values from dataset analysis (NOT zeros!)
    defaults = {
        'setting1': 0.25, 'setting2': -0.0003, 'setting3': 100.0,
        's1': 100.0, 's2': 100.0, 's3': 100.0, 's4': 520.0,
        's5': 1400.0, 's6': 1500.0, 's7': 48.0, 's8': 2800.0,
        's9': 0.65, 's10': 35.0, 's11': 0.72, 's12': 0.90,
        's13': 2300.0, 's14': 3100.0, 's15': 0.40, 's16': 0.45,
        's17': 0.50, 's18': 50.0, 's19': 0.02, 's20': 0.24, 's21': 99.0
    }
    
    base_features = ['setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    for feat in base_features:
        if feat not in df.columns:
            df[feat] = defaults.get(feat, 0.0)
        else:
            # Replace zeros with normal defaults (zeros are unrealistic)
            df[feat] = df[feat].replace(0.0, defaults.get(feat, 0.0))
    
    # Step 2: Compute average features if not present
    if 'temp_avg' not in df.columns:
        df['temp_avg'] = df[['s1', 's2', 's3']].mean(axis=1)
    else:
        df['temp_avg'] = df['temp_avg'].replace(0.0, 100.0)
        
    if 'pressure_avg' not in df.columns:
        df['pressure_avg'] = df[['s7', 's11']].mean(axis=1)
    else:
        df['pressure_avg'] = df['pressure_avg'].replace(0.0, 24.4)
        
    if 'vibration_avg' not in df.columns:
        df['vibration_avg'] = df[['s15', 's16', 's17']].mean(axis=1)
    else:
        df['vibration_avg'] = df['vibration_avg'].replace(0.0, 0.45)
        
    if 'rpm_avg' not in df.columns:
        df['rpm_avg'] = df[['s8', 's9']].mean(axis=1)
    else:
        df['rpm_avg'] = df['rpm_avg'].replace(0.0, 1400.0)
    
    # Step 3: Compute engineered features (same as training)
    df['temp_pressure_ratio'] = df['temp_avg'] / (df['pressure_avg'] + 1)
    df['vibration_rpm_ratio'] = df['vibration_avg'] / (df['rpm_avg'] + 1)
    df['temp_vibration'] = df['temp_avg'] * df['vibration_avg']
    df['pressure_rpm'] = df['pressure_avg'] * df['rpm_avg']
    df['temp_rpm'] = df['temp_avg'] * df['rpm_avg']
    
    # Polynomial features
    df['temp_squared'] = df['temp_avg'] ** 2
    df['vibration_squared'] = df['vibration_avg'] ** 2
    
    # Rolling features (for single prediction, use current values as approximation)
    df['temp_rolling_std'] = df['temp_avg'] * 0.1  # Approximate std
    df['pressure_rolling_std'] = df['pressure_avg'] * 0.1
    df['vibration_rolling_mean'] = df['vibration_avg']
    df['rpm_rolling_max'] = df['rpm_avg'] * 1.1
    df['temp_rolling_mean'] = df['temp_avg']
    
    # Sensor statistics
    sensor_cols = [f's{i}' for i in range(1, 22)]
    sensor_cols = [c for c in sensor_cols if c in df.columns]
    if len(sensor_cols) > 0:
        df['sensor_std'] = df[sensor_cols].std(axis=1)
        df['sensor_range'] = df[sensor_cols].max(axis=1) - df[sensor_cols].min(axis=1)
        df['sensor_mean'] = df[sensor_cols].mean(axis=1)
    else:
        df['sensor_std'] = 0.0
        df['sensor_range'] = 0.0
        df['sensor_mean'] = 0.0
    
    # Step 4: Ensure all model features are present (fill any remaining with 0)
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    
    # Step 5: Reorder columns and convert to numeric
    X = df.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    
    # Fill NaN with 0
    X = X.fillna(0)
    
    return X

@app.post("/predict/compare", tags=["Prediction"])
def predict_compare(body: PredictRequest):
    """Compare predictions from all available models. Stores results in MongoDB if available."""
    if not LOADED_MODELS:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    try:
        X = prepare_input(body.inputs)
        
        comparison = []
        for model_name, bundle in LOADED_MODELS.items():
            model = bundle["pipeline"]
            proba = model.predict_proba(X)[:, 1]
            # Use 0.7 threshold for more conservative predictions (reduces false positives)
            labels = (proba >= 0.7).astype(int)
            
            comparison.append({
                "model": model_name,
                "predictions": ["Needs_Maintenance" if i else "Working" for i in labels],
                "probability": [float(p) for p in proba],
                "confidence": [float(max(p, 1-p)) for p in proba],
                "metrics": bundle.get("metrics", {})
            })
            
            # Store in MongoDB if available
            if mongodb_handler:
                try:
                    prediction_data = {
                        "model": model_name,
                        "prediction": int(labels[0]) if len(labels) > 0 else 0,
                        "confidence": float(max(proba[0], 1-proba[0])) if len(proba) > 0 else 0,
                        "features": body.inputs[0] if body.inputs else {},
                        "model_accuracy": bundle.get("metrics", {}).get("accuracy"),
                        "unit_id": body.inputs[0].get("unit", "unknown") if body.inputs else "unknown"
                    }
                    mongodb_handler.store_prediction(prediction_data)
                except Exception as e:
                    print(f"⚠️  MongoDB storage failed for {model_name}: {e}")
        
        return {
            "comparison": comparison,
            "num_models": len(comparison),
            "mongodb_stored": MONGODB_AVAILABLE and mongodb_handler is not None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")

@app.post("/predict", tags=["Prediction"])
def predict(body: PredictRequest):
    """Make predictions using gradient_boosting as default (best balance of accuracy and practical predictions)."""
    if not LOADED_MODELS:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    # Use gradient_boosting as default (better balance for normal operations)
    # CatBoost and XGBoost are too conservative and predict maintenance even for normal operations
    best_model_name = "gradient_boosting"
    
    # Fallback to other models if gradient_boosting not available
    if best_model_name not in LOADED_MODELS:
        # Try random_forest next
        if "random_forest" in LOADED_MODELS:
            best_model_name = "random_forest"
        else:
            # Last resort: use first available model
            best_model_name = list(LOADED_MODELS.keys())[0]
    
    bundle = LOADED_MODELS[best_model_name]
    model = bundle["pipeline"]
    
    try:
        X = prepare_input(body.inputs)
        
        # Get predictions
        proba = model.predict_proba(X)[:, 1]
        # Use 0.7 threshold for more conservative predictions (reduces false positives)
        labels = (proba >= 0.7).astype(int)
        
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
        # Use 0.7 threshold for more conservative predictions (reduces false positives)
        labels = (proba >= 0.7).astype(int)
        
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
