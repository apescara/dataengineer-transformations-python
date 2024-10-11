from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, udf, round
from pyspark.sql.types import DoubleType
import math

METERS_PER_FOOT = 0.3048
FEET_PER_MILE = 5280
EARTH_RADIUS_IN_METERS = 6371e3
METERS_PER_MILE = METERS_PER_FOOT * FEET_PER_MILE


def get_distance(lat1, lon1, lat2, lon2):
    r1 = lat1 * math.pi / 180
    r2 = lat2 * math.pi / 180
    lambda_r = (lat2 - lat1) * math.pi / 180
    lambda_l = (lon2 - lon1) * math.pi / 180

    a = (math.sin(lambda_r / 2) * math.sin(lambda_r / 2)) + (
        math.cos(r1) * math.cos(r2) * math.sin(lambda_l / 2) * math.sin(lambda_l / 2)
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return (EARTH_RADIUS_IN_METERS * c) / METERS_PER_MILE


def compute_distance(_spark: SparkSession, dataframe: DataFrame) -> DataFrame:
    udf_get_distance = udf(get_distance, DoubleType())
    return dataframe.withColumn(
        "distance",
        round(
            udf_get_distance(
                "start_station_latitude",
                "start_station_longitude",
                "end_station_latitude",
                "end_station_longitude",
            ),
            2,
        ),
    ).sort("end_station_id", ascending = False)


def run(
    spark: SparkSession, input_dataset_path: str, transformed_dataset_path: str
) -> None:
    input_dataset = spark.read.parquet(input_dataset_path)
    input_dataset.show()

    dataset_with_distances = compute_distance(spark, input_dataset)
    dataset_with_distances.show()

    dataset_with_distances.write.parquet(transformed_dataset_path, mode="append")
