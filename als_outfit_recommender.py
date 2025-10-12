import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import pyspark
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType

from recommenders.utils.timer import Timer
from recommenders.utils.notebook_utils import is_jupyter
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.utils.notebook_utils import store_metadata

# Dataset
from datasets import outfits

import matplotlib.pyplot as plt


# top k items to recommend
TOP_K = 1

OUTFITS_DATA_SIZE = '100'

# Column names for the dataset
COL_USER = "UserId"
COL_ITEM = "Clothing"
COL_RATING = "Rating"
COL_WEATHER = "Weather"
COL_ITEM_ID = "ClothingId"

# Start Spark session
spark = start_or_get_spark("ALS PySpark", memory="16g")
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")

schema = StructType(
    (
        StructField(COL_USER, IntegerType()),
        StructField(COL_WEATHER, StringType()),
        StructField(COL_ITEM, StringType()),
        StructField(COL_RATING, FloatType()),
    )
)

data = outfits.load_spark_df(spark, size=None, schema=schema, filepath="./datasets/csv/feature1.csv")

indexer = StringIndexer(inputCol=COL_ITEM, outputCol=COL_ITEM_ID)
indexed_data = indexer.fit(data).transform(data)

train, test = spark_random_split(indexed_data, ratio=0.75, seed=123)

header = {
    "userCol": COL_USER,
    "itemCol": COL_ITEM_ID,
    "ratingCol": COL_RATING,
}


maxIters=15

errors = []

for i in range(1, maxIters + 1):

    als = ALS(
    rank=10,
    maxIter=i,
    implicitPrefs=False,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=False,
    seed=42,
    **header
    )

    with Timer() as train_time:
        model = als.fit(train)


    # with Timer() as test_time:

    #     # Get the cross join of all user-item pairs and score them.
    #     users = train.select(COL_USER).distinct()
    #     items = train.select(COL_ITEM).distinct()
    #     user_item = users.crossJoin(items)
    #     dfs_pred = model.transform(user_item)

    #     # Remove seen items.
    #     dfs_pred_exclude_train = dfs_pred.alias("pred").join(
    #         train.alias("train"),
    #         (dfs_pred[COL_USER] == train[COL_USER]) & (dfs_pred[COL_ITEM] == train[COL_ITEM]),
    #         how='outer'
    #     )

    #     top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train[f"train.{COL_RATING}"].isNull()) \
    #         .select('pred.' + COL_USER, 'pred.' + COL_ITEM, 'pred.' + "prediction")

    #     # In Spark, transformations are lazy evaluation
    #     # Use an action to force execute and measure the test time 
    #     top_all.cache().count()

    # print("Took {} seconds for prediction.".format(test_time.interval))

    with Timer() as test_time:
        users = train.select(COL_USER).distinct()

        items = train.select(COL_ITEM_ID).distinct()

        user_item = users.crossJoin(items)
        dfs_pred = model.transform(user_item)

        top_all = dfs_pred.join(
            indexed_data.select(COL_USER, COL_ITEM_ID),
            on=[COL_USER, COL_ITEM_ID],
            how='left_anti'
        )

        # Force execution to measure the time
        top_all.cache().count()


    rank_eval = SparkRankingEvaluation(test, top_all, k = TOP_K, col_user=COL_USER, col_item=COL_ITEM_ID, 
                                        col_rating=COL_RATING, col_prediction="prediction", 
                                        relevancy_method="top_k")

    # Generate predicted ratings.
    prediction = model.transform(test)

    rating_eval = SparkRatingEvaluation(test, prediction, col_user=COL_USER, col_item=COL_ITEM, 
                                        col_rating=COL_RATING, col_prediction="prediction")

    errors.append(rating_eval.rmse())
    print(f"Iteration {i}: RMSE = {rating_eval.rmse()}")

plt.plot(range(1, maxIters + 1), errors)
plt.xlabel("Training Steps (Iterations)")
plt.ylabel("Error (RMSE)")
plt.title("ALS Model Error Function over Training Steps")
plt.show()
# cleanup spark instance
spark.stop()