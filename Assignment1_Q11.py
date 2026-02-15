from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, regexp_replace, lower, split, size, explode, collect_list, array
from pyspark.sql.types import ArrayType, StringType, FloatType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, RegexTokenizer
from pyspark.ml.linalg import Vectors, VectorUDT
import re
import math

print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("TF-IDF and Book Similarity") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

print("Spark Session created!\n")

# ============================================================================
# STEP 1: Load Books and Extract Filename
# ============================================================================
print("="*80)
print("STEP 1: Loading Books")
print("="*80)

from pyspark.sql.functions import regexp_extract

books_df = spark.read.text("dataset/*.txt", wholetext=True) \
    .selectExpr("input_file_name() as file_path", "value as text")

books_df = books_df.withColumn(
    "file_name", 
    regexp_extract(col("file_path"), r"([^/\\]+\.txt)$", 1)
)

total_books = books_df.count()
print(f"Total books loaded: {total_books}\n")

# ============================================================================
# STEP 2: Preprocessing - Remove Headers/Footers
# ============================================================================
print("="*80)
print("STEP 2: Preprocessing - Removing Project Gutenberg Headers/Footers")
print("="*80)

def remove_gutenberg_header_footer(text):
    """
    Remove Project Gutenberg header and footer from text.
    
    Header typically ends with: "*** START OF THIS PROJECT GUTENBERG EBOOK"
    Footer typically starts with: "*** END OF THIS PROJECT GUTENBERG EBOOK"
    """
    if not text:
        return ""
    
    # Find start marker
    start_markers = [
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK",
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK"
    ]
    
    start_pos = 0
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            # Start after the marker line
            start_pos = text.find('\n', match.end()) + 1
            break
    
    # Find end marker
    end_markers = [
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK",
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK"
    ]
    
    end_pos = len(text)
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            end_pos = match.start()
            break
    
    # Extract clean text
    clean_text = text[start_pos:end_pos]
    
    return clean_text.strip()

remove_header_footer_udf = udf(remove_gutenberg_header_footer, StringType())

books_cleaned = books_df.withColumn("clean_text", remove_header_footer_udf(col("text")))

print("Headers and footers removed.\n")

# ============================================================================
# STEP 3: Text Preprocessing - Lowercase, Remove Punctuation, Tokenize
# ============================================================================
print("="*80)
print("STEP 3: Text Preprocessing")
print("="*80)

# Lowercase
books_cleaned = books_cleaned.withColumn("clean_text", lower(col("clean_text")))
print("✓ Converted to lowercase")

# Remove punctuation and numbers - keep only letters and spaces
books_cleaned = books_cleaned.withColumn(
    "clean_text", 
    regexp_replace(col("clean_text"), r"[^a-z\s]", " ")
)
print("✓ Removed punctuation and numbers")

# Remove extra whitespace
books_cleaned = books_cleaned.withColumn(
    "clean_text", 
    regexp_replace(col("clean_text"), r"\s+", " ")
)
print("✓ Removed extra whitespace")

# Tokenize using RegexTokenizer (better than simple split)
tokenizer = RegexTokenizer(
    inputCol="clean_text", 
    outputCol="words_raw", 
    pattern="\\s+",  # Split on whitespace
    minTokenLength=3  # Remove very short words (< 3 characters)
)

books_tokenized = tokenizer.transform(books_cleaned)
print("✓ Tokenized into words (minimum length: 3 characters)")

# Remove stop words
remover = StopWordsRemover(inputCol="words_raw", outputCol="words")
books_processed = remover.transform(books_tokenized)
print("✓ Removed stop words")

# Show sample
print("\nSample of preprocessed data:")
books_processed.select("file_name", "words").show(5, truncate=80)

# ============================================================================
# STEP 4: Calculate TF-IDF
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Calculating TF-IDF Scores")
print("="*80)

# Calculate Term Frequency (TF) using HashingTF
print("\nCalculating Term Frequency (TF)...")
hashingTF = HashingTF(
    inputCol="words", 
    outputCol="raw_features", 
    numFeatures=10000  # Use 10000 hash buckets
)
books_tf = hashingTF.transform(books_processed)
print("✓ Term Frequency calculated")

# Calculate Inverse Document Frequency (IDF)
print("\nCalculating Inverse Document Frequency (IDF)...")
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(books_tf)
books_tfidf = idfModel.transform(books_tf)
print("✓ IDF calculated and TF-IDF features generated")

# Cache the TF-IDF features for similarity calculation
books_tfidf_cached = books_tfidf.select("file_name", "features").cache()
books_tfidf_cached.count()  # Force cache

print(f"\n✓ TF-IDF vectors cached for {books_tfidf_cached.count()} books")

# ============================================================================
# STEP 5: Calculate Cosine Similarity
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Calculating Cosine Similarity Between Books")
print("="*80)

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two sparse vectors.
    
    Cosine similarity = (A · B) / (||A|| * ||B||)
    Where:
    - A · B is the dot product
    - ||A|| is the magnitude (L2 norm) of vector A
    """
    dot_product = float(vec1.dot(vec2))
    magnitude1 = float(vec1.norm(2))
    magnitude2 = float(vec2.norm(2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

# Register UDF for cosine similarity
cosine_sim_udf = udf(cosine_similarity, FloatType())

# For demonstration, find similar books to "10.txt" (The King James Bible)
target_book = "10.txt"

print(f"\nFinding books similar to: {target_book}")

# Get the feature vector for the target book
target_features = books_tfidf_cached.filter(col("file_name") == target_book).first()

if not target_features:
    print(f"ERROR: Book {target_book} not found!")
    spark.stop()
    exit(1)

target_vector = target_features["features"]
print(f"✓ Retrieved feature vector for {target_book}")

# Broadcast the target vector for efficiency
from pyspark.sql.functions import lit, udf
from pyspark.ml.linalg import VectorUDT

# Create a UDF that calculates similarity to the target
def calc_similarity_to_target(features):
    return cosine_similarity(target_vector, features)

similarity_udf = udf(calc_similarity_to_target, FloatType())

# Calculate similarity for all books
print("\nCalculating cosine similarity for all books...")
books_with_similarity = books_tfidf_cached.withColumn(
    "similarity", 
    similarity_udf(col("features"))
)

# Sort by similarity and get top 6 (including the book itself)
top_similar = books_with_similarity.orderBy(col("similarity").desc()).limit(6)

print(f"\n{'='*80}")
print(f"TOP 5 MOST SIMILAR BOOKS TO: {target_book}")
print(f"{'='*80}\n")

# Show results
result = top_similar.select("file_name", "similarity").collect()

for i, row in enumerate(result):
    if row['file_name'] == target_book:
        print(f"{'[TARGET]':<12} {row['file_name']:<15} Similarity: {row['similarity']:.4f}")
    else:
        print(f"Rank {i:<6} {row['file_name']:<15} Similarity: {row['similarity']:.4f}")

# ============================================================================
# STEP 6: Calculate Similarity for Multiple Books (Optional)
# ============================================================================
print("\n" + "="*80)
print("BONUS: Top 3 Similar Books for Each of First 10 Books")
print("="*80 + "\n")

# Get first 10 books
first_10_books = books_tfidf_cached.limit(10).collect()

for book in first_10_books[:5]:  # Show only 5 for brevity
    book_name = book['file_name']
    book_vector = book['features']
    
    def calc_sim(features):
        return cosine_similarity(book_vector, features)
    
    temp_udf = udf(calc_sim, FloatType())
    
    similar_books = books_tfidf_cached.withColumn("similarity", temp_udf(col("features"))) \
        .orderBy(col("similarity").desc()) \
        .limit(4) \
        .select("file_name", "similarity") \
        .collect()
    
    print(f"\n{book_name}:")
    for i, sim_book in enumerate(similar_books[1:]):  # Skip self
        print(f"  {i+1}. {sim_book['file_name']:<15} (similarity: {sim_book['similarity']:.4f})")

# ============================================================================
# STEP 7: Save Results
# ============================================================================
print("\n" + "="*80)
print("Saving Results...")
print("="*80)

# Save TF-IDF features
books_tfidf.select("file_name", "features").write.parquet(
    "output/tfidf_features.parquet", 
    mode="overwrite"
)
print("✓ TF-IDF features saved to: output/tfidf_features.parquet")

# Save top similar books for target
top_similar.select("file_name", "similarity").coalesce(1).write.csv(
    "output/top_similar_books.csv", 
    header=True, 
    mode="overwrite"
)
print("✓ Top similar books saved to: output/top_similar_books.csv")

print("\n" + "="*80)
print("Question 11 Complete!")
print("="*80)

spark.stop()