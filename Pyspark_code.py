# Q-10
# Import required functions
from pyspark.sql import Row
from pyspark.sql.functions import regexp_extract, col, length

rdd = spark.sparkContext.wholeTextFiles(
    "/mnt/c/Users/ARPITA KUNDU/Downloads/D184MB/D184MB/*.txt"
)


books_df = rdd.map(lambda x: Row(
    file_name=x[0].split("/")[-1],
    text=x[1]
)).toDF()

# Metadata Extraction

metadata_df = books_df \
    .withColumn(
        "title",
        regexp_extract(col("text"), r"(?i)Title:\s*(.+)", 1)
    ) \
    .withColumn(
        "release_date",
        regexp_extract(col("text"), r"(?i)Release Date:\s*(.+)", 1)
    ) \
    .withColumn(
        "language",
        regexp_extract(col("text"), r"(?i)Language:\s*(.+)", 1)
    ) \
    .withColumn(
        "encoding",
        regexp_extract(col("text"), r"(?i)Character set encoding:\s*(.+)", 1)
    )


# Verify extracted metadata
metadata_df.select(
    "file_name", "title", "release_date", "language", "encoding"
).show(10, truncate=False)


# Number of books released each year
books_per_year = metadata_df \
    .filter(col("release_year") != "") \
    .groupBy("release_year") \
    .count() \
    .orderBy("release_year")

books_per_year.show()



# Most common language in the dataset
metadata_df \
    .filter(col("language") != "") \
    .groupBy("language") \
    .count() \
    .orderBy(col("count").desc()) \
    .show(1)


# Average length of book titles (in characters)
metadata_df \
    .filter(col("title") != "") \
    .select(length(col("title")).alias("title_length")) \
    .agg({"title_length": "avg"}) \
    .show()

# Q-11
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# cleaning
clean_df = books_df.withColumn(
    "clean_text",
    lower(regexp_replace(col("text"), "[^a-zA-Z\\s]", " "))
)

# Tokenize
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
tokenized_df = tokenizer.transform(clean_df)

# Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_df = remover.transform(tokenized_df)


#TF-IDF Calculation
from pyspark.ml.feature import HashingTF, IDF

# Term Frequency (TF)
hashingTF = HashingTF(
    inputCol="filtered_words",
    outputCol="raw_features",
    numFeatures=10000
)

featurized_df = hashingTF.transform(filtered_df)

# Inverse Document Frequency (IDF)

idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(featurized_df)

tfidf_df = idf_model.transform(featurized_df)

# Cosine Similarity
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import numpy as np

def cosine_similarity(v1, v2):
    dot = float(v1.dot(v2))
    norm1 = float(np.linalg.norm(v1.toArray()))
    norm2 = float(np.linalg.norm(v2.toArray()))
    return float(dot / (norm1 * norm2)) if norm1 != 0 and norm2 != 0 else 0.0

cosine_udf = udf(cosine_similarity, DoubleType())


# Q-12

# Extract Author + Release Date
from pyspark.sql.functions import regexp_extract, col

author_df = books_df \
    .withColumn(
        "author",
        regexp_extract(col("text"), r"(?i)Author:\s*(.+)", 1)
    ) \
    .withColumn(
        "release_date",
        regexp_extract(col("text"), r"(?i)Release Date:\s*(.+)", 1)
    )

# Extract Release Year
author_df = author_df.withColumn(
    "release_year",
    regexp_extract(col("release_date"), r"(\d{4})", 1)
)

author_df = author_df.filter(
    (col("author") != "") &
    (col("release_year") != "")
)

# Construct Influence Network
X = 5
# Self-Join to Compare Authors
from pyspark.sql.functions import abs

a = author_df.alias("a")
b = author_df.alias("b")

edges_df = a.join(b, col("a.author") != col("b.author")) \
    .filter(
        (col("b.release_year").cast("int") > col("a.release_year").cast("int")) &
        ((col("b.release_year").cast("int") - col("a.release_year").cast("int")) <= X)
    ) \
    .select(
        col("a.author").alias("author1"),
        col("b.author").alias("author2")
    )


# Compute In-Degree and Out-Degree

# Out-Degree (how many influenced)
out_degree = edges_df.groupBy("author1") \
    .count() \
    .withColumnRenamed("count", "out_degree")

# In-Degree (how many influenced them)
in_degree = edges_df.groupBy("author2") \
    .count() \
    .withColumnRenamed("count", "in_degree")

# Top 5 Authors

# Top 5 by Out-Degre
out_degree.orderBy(col("out_degree").desc()).show(5)

# Top 5 by In-Degree
in_degree.orderBy(col("in_degree").desc()).show(5)
