"""
Apache Spark (PySpark) Integration for Big Data Processing
Handles large-scale data processing and feature engineering
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, max as spark_max, min as spark_min
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pandas as pd
import os

class SparkDataProcessor:
    """Apache Spark processor for big data operations"""
    
    def __init__(self, app_name: str = "PredictiveMaintenance"):
        """Initialize Spark session"""
        try:
            # type: ignore is needed for PySpark's dynamic builder pattern
            self.spark = SparkSession.builder.appName(app_name).config(  # type: ignore
                "spark.driver.memory", "2g"
            ).config(
                "spark.executor.memory", "2g"
            ).config(
                "spark.sql.adaptive.enabled", "true"
            ).getOrCreate()
            
            # Set log level to reduce verbosity
            self.spark.sparkContext.setLogLevel("ERROR")
            print("✅ Apache Spark initialized successfully")
            print(f"   Spark Version: {self.spark.version}")
            
        except Exception as e:
            print(f"⚠️  Spark initialization failed: {e}")
            self.spark = None
    
    def load_csv_data(self, file_path):
        """
        Load CSV data using Spark for distributed processing
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            DataFrame: Spark DataFrame
        """
        if not self.spark:
            raise RuntimeError("Spark not initialized")
        
        print(f"📊 Loading data with Apache Spark: {file_path}")
        df = self.spark.read.csv(file_path, header=True, inferSchema=True)
        
        print(f"✅ Loaded {df.count()} rows with {len(df.columns)} columns")
        return df
    
    def compute_statistics(self, df, columns):
        """
        Compute statistical aggregations using Spark
        
        Args:
            df: Spark DataFrame
            columns (list): Columns to analyze
            
        Returns:
            dict: Statistics for each column
        """
        if not self.spark:
            return {}
        
        stats = {}
        for column in columns:
            agg_df = df.agg(
                avg(col(column)).alias('mean'),
                stddev(col(column)).alias('stddev'),
                spark_min(col(column)).alias('min'),
                spark_max(col(column)).alias('max')
            ).collect()[0]
            
            stats[column] = {
                'mean': float(agg_df['mean']) if agg_df['mean'] else 0,
                'stddev': float(agg_df['stddev']) if agg_df['stddev'] else 0,
                'min': float(agg_df['min']) if agg_df['min'] else 0,
                'max': float(agg_df['max']) if agg_df['max'] else 0
            }
        
        return stats
    
    def feature_engineering_spark(self, df, sensor_columns):
        """
        Perform feature engineering using Spark
        
        Args:
            df: Spark DataFrame
            sensor_columns (list): List of sensor column names
            
        Returns:
            DataFrame: Spark DataFrame with engineered features
        """
        if not self.spark:
            return df
        
        print("🔧 Performing feature engineering with Apache Spark...")
        
        # Create aggregated features
        for prefix in ['temp', 'pressure', 'vibration', 'rpm']:
            matching_cols = [c for c in sensor_columns if prefix in c.lower()]
            if matching_cols:
                # Calculate average for this feature group
                avg_col = sum([col(c) for c in matching_cols]) / len(matching_cols)
                df = df.withColumn(f"{prefix}_avg", avg_col)
        
        print("✅ Feature engineering completed")
        return df
    
    def convert_to_pandas(self, spark_df):
        """
        Convert Spark DataFrame to Pandas for ML processing
        
        Args:
            spark_df: Spark DataFrame
            
        Returns:
            DataFrame: Pandas DataFrame
        """
        if not self.spark:
            return None
        
        print("🔄 Converting Spark DataFrame to Pandas...")
        pandas_df = spark_df.toPandas()
        print(f"✅ Converted to Pandas: {len(pandas_df)} rows")
        return pandas_df
    
    def process_large_dataset(self, file_path):
        """
        Complete processing pipeline using Spark
        
        Args:
            file_path (str): Path to dataset
            
        Returns:
            DataFrame: Processed Pandas DataFrame ready for ML
        """
        if not self.spark:
            print("⚠️  Spark not available, falling back to Pandas")
            return pd.read_csv(file_path)
        
        try:
            # Load with Spark
            spark_df = self.load_csv_data(file_path)
            
            # Get sensor columns
            sensor_columns = [c for c in spark_df.columns if c.startswith('s')]
            
            # Feature engineering
            spark_df = self.feature_engineering_spark(spark_df, sensor_columns)
            
            # Convert to Pandas for ML
            pandas_df = self.convert_to_pandas(spark_df)
            
            print("✅ Big Data processing completed with Apache Spark")
            return pandas_df
            
        except Exception as e:
            print(f"⚠️  Spark processing failed: {e}")
            print("   Falling back to Pandas processing")
            return pd.read_csv(file_path)
    
    def get_data_insights(self, file_path):
        """
        Get quick data insights using Spark
        
        Args:
            file_path (str): Path to dataset
            
        Returns:
            dict: Data insights
        """
        if not self.spark:
            return {"status": "Spark not available"}
        
        try:
            df = self.load_csv_data(file_path)
            
            insights = {
                "total_rows": df.count(),
                "total_columns": len(df.columns),
                "column_names": df.columns,
                "data_types": {field.name: str(field.dataType) for field in df.schema.fields}
            }
            
            # Get sample statistics for numeric columns
            numeric_cols = [field.name for field in df.schema.fields 
                          if 'Int' in str(field.dataType) or 'Double' in str(field.dataType)]
            
            if numeric_cols[:5]:  # Limit to first 5 for performance
                insights["statistics"] = self.compute_statistics(df, numeric_cols[:5])
            
            return insights
            
        except Exception as e:
            return {"error": str(e)}
    
    def stop(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            print("🛑 Spark session stopped")

# Global Spark processor instance
spark_processor = None

def get_spark_processor():
    """Get or create Spark processor instance"""
    global spark_processor
    if spark_processor is None:
        try:
            spark_processor = SparkDataProcessor()
        except Exception as e:
            print(f"⚠️  Spark not available: {e}")
            print("   Will use Pandas for data processing (optional feature)")
            spark_processor = None
    return spark_processor
