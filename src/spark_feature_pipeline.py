from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


spark = SparkSession.builder \
    .appName("Enterprise Customer Feature Pipeline") \
    .getOrCreate()


df = spark.read.csv(
    "data/processed/customer_data.csv",
    header=True,
    inferSchema=True
)


feature_df = df.withColumn(
    "engagement_segment",
    when(col("monthly_active_days") < 10, "Low Engagement")
    .when(col("monthly_active_days") < 20, "Medium Engagement")
    .otherwise("High Engagement")
).withColumn(
    "sentiment_risk_segment",
    when(col("negative_sentiment_ratio") > 0.7, "High Risk")
    .when(col("negative_sentiment_ratio") > 0.4, "Medium Risk")
    .otherwise("Low Risk")
).withColumn(
    "payment_risk_segment",
    when(col("payment_delay_count") >= 3, "High Payment Risk")
    .when(col("payment_delay_count") >= 1, "Medium Payment Risk")
    .otherwise("Low Payment Risk")
)


feature_df.write.mode("overwrite").csv(
    "data/processed/spark_customer_features",
    header=True
)

print("Spark feature pipeline completed successfully")

spark.stop()