"""
MongoDB Configuration for Big Data Storage
Stores prediction history and analytics data
"""
from pymongo import MongoClient
from datetime import datetime
import os

class MongoDBHandler:
    """Handle MongoDB operations for prediction storage"""
    
    def __init__(self, connection_string="mongodb://localhost:27017/"):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=2000)
            self.db = self.client['predictive_maintenance']
            self.predictions_collection = self.db['predictions']
            self.analytics_collection = self.db['analytics']
            
            # Test connection
            self.client.admin.command('ping')
            
            # Create indexes for better query performance
            self.predictions_collection.create_index([("timestamp", -1)])
            self.predictions_collection.create_index([("model_name", 1)])
            self.predictions_collection.create_index([("unit_id", 1)])
            
        except Exception as e:
            raise ConnectionError(f"Could not connect to MongoDB: {e}")
        
    def store_prediction(self, prediction_data):
        """
        Store prediction result in MongoDB
        
        Args:
            prediction_data (dict): Prediction details including model, result, confidence
        
        Returns:
            str: Inserted document ID
        """
        document = {
            "timestamp": datetime.utcnow(),
            "model_name": prediction_data.get("model", "unknown"),
            "prediction": prediction_data.get("prediction", 0),
            "confidence": prediction_data.get("confidence", 0.0),
            "sensor_data": prediction_data.get("features", {}),
            "unit_id": prediction_data.get("unit_id", "default"),
            "prediction_label": "Maintenance Required" if prediction_data.get("prediction") == 1 else "Normal Operation",
            "metadata": {
                "accuracy": prediction_data.get("model_accuracy"),
                "processing_time_ms": prediction_data.get("processing_time")
            }
        }
        
        result = self.predictions_collection.insert_one(document)
        return str(result.inserted_id)
    
    def get_recent_predictions(self, limit=100):
        """Get recent predictions from MongoDB"""
        cursor = self.predictions_collection.find().sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    def get_predictions_by_model(self, model_name):
        """Get all predictions for a specific model"""
        cursor = self.predictions_collection.find({"model_name": model_name})
        return list(cursor)
    
    def get_statistics(self):
        """Get prediction statistics"""
        total_predictions = self.predictions_collection.count_documents({})
        maintenance_predictions = self.predictions_collection.count_documents({"prediction": 1})
        
        # Get model usage statistics
        pipeline = [
            {"$group": {"_id": "$model_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        model_stats = list(self.predictions_collection.aggregate(pipeline))
        
        return {
            "total_predictions": total_predictions,
            "maintenance_predictions": maintenance_predictions,
            "normal_predictions": total_predictions - maintenance_predictions,
            "model_usage": model_stats
        }
    
    def clear_old_predictions(self, days=30):
        """Remove predictions older than specified days"""
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        result = self.predictions_collection.delete_many({"timestamp": {"$lt": cutoff_date}})
        return result.deleted_count
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()

# Global MongoDB handler instance
mongodb_handler = None

def get_mongodb_handler():
    """Get or create MongoDB handler instance"""
    global mongodb_handler
    if mongodb_handler is None:
        try:
            mongodb_handler = MongoDBHandler()
            print("✅ MongoDB connected successfully")
        except Exception as e:
            print(f"⚠️  MongoDB not available: {e}")
            print("   Predictions will not be stored (optional feature)")
            mongodb_handler = None
    return mongodb_handler
