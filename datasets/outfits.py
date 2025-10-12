# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pandas as pd
import warnings
from typing import Optional

from recommenders.datasets.download_utils import download_path
from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        IntegerType,
        FloatType,
    )
except ImportError:
    pass  # so the environment without spark doesn't break

import pandera as pa
from pandera import Field
from pandera.typing import Series


# Fake data for testing only
MOCK_OUTFIT_FORMAT = {
    "mock100": {"size": 100, "seed": 123},
}

# Warning and error messages
WARNING_OUTFIT_DATASET_HEADER = """Outfit dataset has four columns
    (user id, weather, clothing, and rating), but more than four column names are provided.
    Will only use the first four column names."""
ERROR_HEADER = "Header error. At least user and clothing column names should be provided"


def load_pandas_df(
    filepath="example.csv",
    header=None,
):
    """Loads the Outfit dataset as pd.DataFrame.

    Args:
        filepath (str): Path to the outfit dataset CSV file.
        header (list or tuple or None): Rating dataset header.
        weather_col (str): Weather column name. If None, the column will not be loaded.

    Returns:
        pandas.DataFrame: Outfit rating dataset.
    """
    if header is None:
        header = [DEFAULT_USER_COL, "Weather", DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
    elif len(header) < 2:
        raise ValueError(ERROR_HEADER)
    elif len(header) > 4:
        warnings.warn(WARNING_OUTFIT_DATASET_HEADER)
        header = header[:4]

    df = pd.read_csv(
        filepath,
        header=0,
        names=header,
    )

    # Convert 'rating' type to float
    if len(header) > 3:
        df[header[3]] = df[header[3]].astype(float)

    return df


def load_spark_df(
    spark,
    size,
    filepath="outfit_data.csv",
    schema=None,
):
    """Loads the Outfit dataset as `pyspark.sql.DataFrame`.

    Args:
        spark (pyspark.SparkSession): Spark session.
        size (str, optional): Size of the data to load. If a key from MOCK_OUTFIT_FORMAT
            (e.g., "mock100"), mock data will be generated. Otherwise, loads from `filepath`.
            Defaults to None.
        filepath (str): Path to the outfit dataset CSV file. Ignored if `size` is a mock size.
        schema (pyspark.StructType): Dataset schema.
        weather_col (str): Weather column name. If None, the column will not be loaded.

    Returns:
        pyspark.sql.DataFrame: Outfit rating dataset.
    """
    if size is not None and size.lower() in MOCK_OUTFIT_FORMAT:
        size_key = size.lower()
        # generate fake data
        return MockOutfitSchema.get_spark_df(
            spark,
            **MOCK_OUTFIT_FORMAT[size_key]  # supply the rest of the kwarg with the dictionary
        )

    if schema is None:
        schema = StructType(
            [
                StructField(DEFAULT_USER_COL, IntegerType()),
                StructField("Weather", StringType()),
                StructField(DEFAULT_ITEM_COL, StringType()),
                StructField(DEFAULT_RATING_COL, FloatType()),
            ]
        )

    df = spark.read.csv(
        filepath,
        schema=schema,
        header=True,
    )

    # Cache and force trigger action
    # df.cache()
    df.count()

    return df


class MockOutfitSchema(pa.DataFrameModel):
    """
    Mock dataset schema to generate fake data for testing purpose.
    This schema is configured to mimic the Outfit dataset.
    """

    userID: Series[int] = Field(
        in_range={"min_value": 1, "max_value": 50}, alias=DEFAULT_USER_COL
    )
    weather: Series[str] = Field(isin=["Sunny", "Cloudy", "Rainy", "Snowy", "Humid", "Windy"])
    clothing: Series[str] = Field(
        isin=[
            "Hoodie",
            "Jeans",
            "T-shirt",
            "Shorts",
            "Coat",
            "Blazer",
            "Cardigan",
            "Joggers",
            "Long-sleeve shirt",
            "Polo",
            "Chinos",
            "Sweatshirt",
            "Dress pants",
        ], alias=DEFAULT_ITEM_COL
    )
    rating: Series[float] = Field(
        in_range={"min_value": 1, "max_value": 5}, alias=DEFAULT_RATING_COL
    )

    class Config:
        # Ensures the column alias (e.g., 'userID') is used in the DataFrame
        name = "MockOutfitSchema"
        coerce = True


    @classmethod
    def get_df(
        cls,
        size: int = 3,
        seed: int = 100,
    ) -> pd.DataFrame:
        """Return fake outfit dataset as a Pandas Dataframe with specified rows.

        Args:
            size (int): number of rows to generate
            seed (int, optional): seeding the pseudo-number generation. Defaults to 100.

        Returns:
            pandas.DataFrame: a mock dataset
        """
        return cls.to_schema().example(size=size, random_state=seed)

    @classmethod
    def get_spark_df(
        cls,
        spark,
        size: int = 3,
        seed: int = 100,
        tmp_path: Optional[str] = None,
    ):
        """Return fake outfit dataset as a Spark Dataframe with specified rows

        Args:
            spark (SparkSession): spark session to load the dataframe into
            size (int): number of rows to generate
            seed (int): seeding the pseudo-number generation. Defaults to 100.
            tmp_path (str, optional): path to store files for serialization purpose
                when transferring data from python to java.
                If None, a temporal path is used instead

        Returns:
            pyspark.sql.DataFrame: a mock dataset
        """
        pandas_df = cls.get_df(size=size, seed=seed)

        # generate temp folder
        with download_path(tmp_path) as tmp_folder:
            filepath = os.path.join(tmp_folder, f"mock_outfit_{size}.csv")
            # serialize the pandas.df as a csv
            pandas_df.to_csv(filepath, header=True, index=False)
            spark_df = spark.read.csv(
                filepath, schema=cls._get_spark_deserialization_schema(), header=True
            )
            # Cache and force trigger action since data-file might be removed.
            spark_df.cache()
            spark_df.count()

        return spark_df

    @classmethod
    def _get_spark_deserialization_schema(cls):
        return StructType(
            [
                StructField(DEFAULT_USER_COL, IntegerType()),
                StructField("Weather", StringType()),
                StructField(DEFAULT_ITEM_COL, StringType()),
                StructField(DEFAULT_RATING_COL, FloatType()),
            ]
        )