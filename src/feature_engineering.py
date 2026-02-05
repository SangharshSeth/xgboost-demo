"""
PySpark Feature Engineering for Fraud Detection

Creates features from raw transaction data for XGBoost model training.
Features include:
- Temporal features (hour, day, weekend, time since last txn)
- Velocity features (transaction count/amount in windows)
- Behavioral features (deviation from normal patterns)
- Balance features (utilization, changes)
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, BooleanType
import os


def get_spark_session() -> SparkSession:
    """Create or get Spark session configured for local execution."""
    return (SparkSession.builder
            .appName("FraudDetection-FeatureEngineering")
            .master("local[*]")  # Use all available cores
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.adaptive.enabled", "true")
            # Fix for Java 17+ SecurityManager deprecation error
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
            .getOrCreate())


def load_data(spark: SparkSession, data_dir: str = "data") -> tuple[DataFrame, DataFrame]:
    """Load transactions and customers data."""
    transactions_path = os.path.join(data_dir, "transactions.parquet")
    customers_path = os.path.join(data_dir, "customers.parquet")
    
    transactions_df = spark.read.parquet(transactions_path)
    customers_df = spark.read.parquet(customers_path)
    
    return transactions_df, customers_df


def add_temporal_features(df: DataFrame) -> DataFrame:
    """Add time-based features."""
    df = df.withColumn("hour_of_day", F.hour("timestamp"))
    df = df.withColumn("day_of_week", F.dayofweek("timestamp"))  # 1=Sunday, 7=Saturday
    df = df.withColumn("day_of_month", F.dayofmonth("timestamp"))
    df = df.withColumn("month", F.month("timestamp"))
    df = df.withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))
    
    # Is it night time? (potential fraud indicator)
    df = df.withColumn("is_night", 
                       F.when((F.col("hour_of_day") >= 0) & (F.col("hour_of_day") < 6), 1).otherwise(0))
    
    # Is it salary week? (1st week of month - more spending expected)
    df = df.withColumn("is_salary_week",
                       F.when(F.col("day_of_month") <= 7, 1).otherwise(0))
    
    # Timestamp as Unix epoch for calculations
    df = df.withColumn("timestamp_epoch", F.unix_timestamp("timestamp"))
    
    return df


def add_lag_features(df: DataFrame) -> DataFrame:
    """Add features based on previous transactions for each customer."""
    # Window for lag calculations (ordered by time)
    customer_window = Window.partitionBy("customer_id").orderBy("timestamp")
    
    # Time since last transaction (in seconds)
    df = df.withColumn("prev_timestamp", F.lag("timestamp_epoch").over(customer_window))
    df = df.withColumn("time_since_last_txn", 
                       F.when(F.col("prev_timestamp").isNull(), 0)
                       .otherwise(F.col("timestamp_epoch") - F.col("prev_timestamp")))
    
    # Previous transaction amount
    df = df.withColumn("prev_amount", F.lag("amount").over(customer_window))
    df = df.withColumn("prev_amount", F.coalesce(F.col("prev_amount"), F.lit(0)))
    
    # Amount change from last transaction
    df = df.withColumn("amount_change", 
                       F.when(F.col("prev_amount") > 0, 
                              (F.col("amount") - F.col("prev_amount")) / F.col("prev_amount"))
                       .otherwise(0))
    
    # Previous city (to detect location jumps)
    df = df.withColumn("prev_city", F.lag("city").over(customer_window))
    df = df.withColumn("is_city_change", 
                       F.when(F.col("prev_city").isNull(), 0)
                       .when(F.col("city") != F.col("prev_city"), 1)
                       .otherwise(0))
    
    # Previous merchant
    df = df.withColumn("prev_merchant", F.lag("merchant_name").over(customer_window))
    df = df.withColumn("is_same_merchant", 
                       F.when(F.col("prev_merchant") == F.col("merchant_name"), 1).otherwise(0))
    
    # Transaction sequence number for customer
    df = df.withColumn("txn_sequence", F.row_number().over(customer_window))
    
    # Drop temporary columns
    df = df.drop("prev_timestamp", "prev_city", "prev_merchant")
    
    return df


def add_velocity_features(df: DataFrame) -> DataFrame:
    """Add rolling window velocity features."""
    # Windows for different time periods (in seconds)
    one_hour = 3600
    six_hours = 6 * 3600
    one_day = 24 * 3600
    seven_days = 7 * 24 * 3600
    thirty_days = 30 * 24 * 3600
    
    def create_range_window(seconds_back):
        return (Window.partitionBy("customer_id")
                .orderBy(F.col("timestamp_epoch"))
                .rangeBetween(-seconds_back, 0))
    
    # Transaction count in windows
    df = df.withColumn("txn_count_1h", F.count("*").over(create_range_window(one_hour)))
    df = df.withColumn("txn_count_6h", F.count("*").over(create_range_window(six_hours)))
    df = df.withColumn("txn_count_24h", F.count("*").over(create_range_window(one_day)))
    df = df.withColumn("txn_count_7d", F.count("*").over(create_range_window(seven_days)))
    
    # Amount sum in windows
    df = df.withColumn("amount_sum_1h", F.sum("amount").over(create_range_window(one_hour)))
    df = df.withColumn("amount_sum_6h", F.sum("amount").over(create_range_window(six_hours)))
    df = df.withColumn("amount_sum_24h", F.sum("amount").over(create_range_window(one_day)))
    df = df.withColumn("amount_sum_7d", F.sum("amount").over(create_range_window(seven_days)))
    
    # Average amount in windows
    df = df.withColumn("amount_avg_7d", F.avg("amount").over(create_range_window(seven_days)))
    df = df.withColumn("amount_avg_30d", F.avg("amount").over(create_range_window(thirty_days)))
    
    # Max amount in windows
    df = df.withColumn("amount_max_7d", F.max("amount").over(create_range_window(seven_days)))
    df = df.withColumn("amount_max_30d", F.max("amount").over(create_range_window(thirty_days)))
    
    # Unique merchants in last 7 days
    df = df.withColumn("unique_merchants_7d", 
                       F.size(F.collect_set("merchant_name").over(create_range_window(seven_days))))
    
    # Unique cities in last 7 days
    df = df.withColumn("unique_cities_7d",
                       F.size(F.collect_set("city").over(create_range_window(seven_days))))
    
    return df


def add_behavioral_features(df: DataFrame) -> DataFrame:
    """Add features comparing current transaction to customer's normal behavior."""
    # Calculate customer-level statistics
    customer_stats = df.groupBy("customer_id").agg(
        F.avg("amount").alias("customer_avg_amount"),
        F.stddev("amount").alias("customer_std_amount"),
        F.count("*").alias("customer_total_txns"),
        F.avg("hour_of_day").alias("customer_avg_hour"),
        F.stddev("hour_of_day").alias("customer_std_hour")
    )
    
    # Fill null std with 1 (for customers with single transaction)
    customer_stats = customer_stats.fillna(1.0, subset=["customer_std_amount", "customer_std_hour"])
    
    # Join back
    df = df.join(customer_stats, on="customer_id", how="left")
    
    # Z-score of amount (how unusual is this amount for this customer)
    df = df.withColumn("amount_zscore", 
                       F.when(F.col("customer_std_amount") > 0,
                              (F.col("amount") - F.col("customer_avg_amount")) / F.col("customer_std_amount"))
                       .otherwise(0))
    
    # Is amount unusually high? (> 2 std deviations)
    df = df.withColumn("is_high_amount", 
                       F.when(F.col("amount_zscore") > 2, 1).otherwise(0))
    
    # Is amount unusually low? (< -2 std deviations)
    df = df.withColumn("is_low_amount",
                       F.when(F.col("amount_zscore") < -2, 1).otherwise(0))
    
    # Hour z-score (is this an unusual time for this customer?)
    df = df.withColumn("hour_zscore",
                       F.when(F.col("customer_std_hour") > 0,
                              (F.col("hour_of_day") - F.col("customer_avg_hour")) / F.col("customer_std_hour"))
                       .otherwise(0))
    
    # Amount compared to 7-day average
    df = df.withColumn("amount_vs_avg_7d",
                       F.when(F.col("amount_avg_7d") > 0,
                              F.col("amount") / F.col("amount_avg_7d"))
                       .otherwise(1))
    
    # Amount compared to 30-day max
    df = df.withColumn("amount_vs_max_30d",
                       F.when(F.col("amount_max_30d") > 0,
                              F.col("amount") / F.col("amount_max_30d"))
                       .otherwise(1))
    
    # Drop intermediate columns
    df = df.drop("customer_std_amount", "customer_avg_amount", 
                 "customer_std_hour", "customer_avg_hour")
    
    return df


def add_balance_features(df: DataFrame) -> DataFrame:
    """Add features related to account balance."""
    # Balance utilization (how much of initial balance is used)
    # We need to join with customer initial balance
    # For now, use balance_after directly
    
    # Balance change percentage
    df = df.withColumn("balance_change_pct",
                       F.when(F.col("balance_before") > 0,
                              (F.col("balance_after") - F.col("balance_before")) / F.col("balance_before"))
                       .otherwise(0))
    
    # Is this transaction using significant portion of balance?
    df = df.withColumn("balance_impact_ratio",
                       F.when(F.col("balance_before") > 0,
                              F.col("amount") / F.col("balance_before"))
                       .otherwise(0))
    
    # Is balance going low? (< 10% of a threshold, say 10000)
    df = df.withColumn("is_low_balance",
                       F.when(F.col("balance_after") < 10000, 1).otherwise(0))
    
    # Is this a large withdrawal relative to balance?
    df = df.withColumn("is_large_withdrawal",
                       F.when((F.col("is_credit") == False) & 
                              (F.col("balance_impact_ratio") > 0.3), 1).otherwise(0))
    
    return df


def add_merchant_features(df: DataFrame) -> DataFrame:
    """Add merchant and category level features."""
    # Count transactions per merchant (merchant popularity)
    merchant_stats = df.groupBy("merchant_name").agg(
        F.count("*").alias("merchant_txn_count"),
        F.avg("amount").alias("merchant_avg_amount"),
        F.sum(F.when(F.col("is_fraud") == True, 1).otherwise(0)).alias("merchant_fraud_count")
    )
    
    # Join back (avoiding label leakage - in production this would use historical data)
    df = df.join(merchant_stats, on="merchant_name", how="left")
    
    # Is this a new/rare merchant for the customer?
    customer_merchant_window = Window.partitionBy("customer_id", "merchant_name").orderBy("timestamp")
    df = df.withColumn("is_first_merchant_txn",
                       F.when(F.row_number().over(customer_merchant_window) == 1, 1).otherwise(0))
    
    # Category encoding (we'll use target encoding in training)
    # For now, create a simple risk score based on category
    category_risk = {
        "grocery": 0.1, "food_delivery": 0.15, "ecommerce": 0.3,
        "fuel": 0.15, "utilities": 0.05, "travel": 0.25,
        "entertainment": 0.2, "healthcare": 0.1, "education": 0.05,
        "rent": 0.05, "emi": 0.05, "investment": 0.3, "p2p_transfer": 0.35,
        "income": 0.01
    }
    
    # Create category risk mapping
    category_risk_expr = F.lit(0.2)  # Default risk
    for cat, risk in category_risk.items():
        category_risk_expr = F.when(F.col("merchant_category") == cat, F.lit(risk)).otherwise(category_risk_expr)
    
    df = df.withColumn("category_risk_score", category_risk_expr)
    
    return df


def encode_categorical_features(df: DataFrame) -> DataFrame:
    """Encode categorical features as integers for XGBoost."""
    # Transaction type encoding
    txn_type_map = {
        "UPI": 1, "DEBIT_CARD": 2, "CREDIT_CARD": 3, "NEFT": 4,
        "IMPS": 5, "ATM": 6, "SALARY": 7, "PENSION": 8, "EMI": 9,
        "REFUND": 10, "PARENT_TRANSFER": 11
    }
    
    txn_type_expr = F.lit(0)
    for txn_type, code in txn_type_map.items():
        txn_type_expr = F.when(F.col("transaction_type") == txn_type, F.lit(code)).otherwise(txn_type_expr)
    df = df.withColumn("transaction_type_encoded", txn_type_expr)
    
    # Status encoding
    df = df.withColumn("status_encoded",
                       F.when(F.col("status") == "SUCCESS", 1)
                       .when(F.col("status") == "FAILED", 0)
                       .otherwise(2))  # PENDING
    
    # City tier encoding
    df = df.withColumn("city_tier_encoded",
                       F.when(F.col("city").isin(["Mumbai", "Delhi", "Bangalore", "Chennai", 
                                                   "Hyderabad", "Kolkata", "Pune", "Ahmedabad"]), 1)
                       .when(F.col("city").isin(["Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore",
                                                  "Bhopal", "Visakhapatnam", "Patna", "Vadodara",
                                                  "Coimbatore", "Ludhiana", "Agra", "Nashik",
                                                  "Ranchi", "Guwahati"]), 2)
                       .otherwise(3))
    
    # Is credit encoding
    df = df.withColumn("is_credit_encoded", F.col("is_credit").cast("integer"))
    
    return df


def select_features(df: DataFrame) -> DataFrame:
    """Select final features for model training."""
    feature_columns = [
        # Identifiers (for reference, not for training)
        "transaction_id",
        "customer_id",
        "timestamp",
        
        # ========== FEATURES ==========
        # Amount features
        "amount",
        "amount_zscore",
        "is_high_amount",
        "is_low_amount",
        "amount_change",
        "amount_vs_avg_7d",
        "amount_vs_max_30d",
        
        # Temporal features
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "is_weekend",
        "is_night",
        "is_salary_week",
        
        # Lag features
        "time_since_last_txn",
        "is_city_change",
        "is_same_merchant",
        "txn_sequence",
        
        # Velocity features
        "txn_count_1h",
        "txn_count_6h",
        "txn_count_24h",
        "txn_count_7d",
        "amount_sum_1h",
        "amount_sum_6h",
        "amount_sum_24h",
        "amount_sum_7d",
        "unique_merchants_7d",
        "unique_cities_7d",
        
        # Balance features
        "balance_before",
        "balance_after",
        "balance_change_pct",
        "balance_impact_ratio",
        "is_low_balance",
        "is_large_withdrawal",
        
        # Merchant features
        "merchant_txn_count",
        "merchant_avg_amount",
        "is_first_merchant_txn",
        "category_risk_score",
        
        # Customer features
        "customer_total_txns",
        
        # Encoded categoricals
        "transaction_type_encoded",
        "status_encoded",
        "city_tier_encoded",
        "is_credit_encoded",
        
        # ========== TARGET ==========
        "is_fraud"
    ]
    
    return df.select(*feature_columns)


def run_feature_engineering(data_dir: str = "data", output_dir: str = "data") -> None:
    """Run the complete feature engineering pipeline."""
    print("=" * 60)
    print("PySpark Feature Engineering Pipeline")
    print("=" * 60)
    
    # Initialize Spark
    print("\n[1/8] Initializing Spark session...")
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Load data
    print("[2/8] Loading transaction data...")
    transactions_df, customers_df = load_data(spark, data_dir)
    print(f"Loaded {transactions_df.count()} transactions")
    
    # Add features step by step
    print("[3/8] Adding temporal features...")
    df = add_temporal_features(transactions_df)
    
    print("[4/8] Adding lag features...")
    df = add_lag_features(df)
    
    print("[5/8] Adding velocity features...")
    df = add_velocity_features(df)
    
    print("[6/8] Adding behavioral features...")
    df = add_behavioral_features(df)
    
    print("[7/8] Adding balance and merchant features...")
    df = add_balance_features(df)
    df = add_merchant_features(df)
    df = encode_categorical_features(df)
    
    # Select final features
    print("[8/8] Selecting features and saving...")
    df = select_features(df)
    
    # Fill any remaining nulls
    df = df.fillna(0)
    
    # Save
    output_path = os.path.join(output_dir, "features.parquet")
    df.write.mode("overwrite").parquet(output_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"\nTotal records: {df.count()}")
    print(f"Total features: {len(df.columns) - 4}")  # Exclude id, customer_id, timestamp, target
    print(f"Fraud cases: {df.filter(F.col('is_fraud') == True).count()}")
    print(f"\nOutput saved to: {output_path}")
    
    # Show sample
    print("\nSample of features:")
    df.select("transaction_id", "amount", "amount_zscore", "txn_count_24h", 
              "is_city_change", "is_fraud").show(5, truncate=False)
    
    spark.stop()


if __name__ == "__main__":
    run_feature_engineering()
