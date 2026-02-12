# Apache Hadoop & Apache Spark

## Assignment 1 -- MapReduce and Spark Analysis

**Name:** Arpita Kundu\
**Roll No:** M25DE1004

------------------------------------------------------------------------

# 📌 Project Overview

This assignment demonstrates practical implementation of:

-   Apache Hadoop MapReduce (WordCount)
-   HDFS operations
-   Performance tuning using split size
-   Apache Spark DataFrame processing
-   Metadata extraction using regex
-   TF-IDF computation
-   Cosine similarity for document similarity
-   Author Influence Network construction

------------------------------------------------------------------------

# 🟢 PART 1 -- Apache Hadoop & MapReduce

## 1️⃣ WordCount Execution

The Hadoop WordCount example was successfully executed using:

hdfs dfs -mkdir -p /lyrics/input\
hdfs dfs -put wc1.txt /lyrics/input

hadoop jar
\$HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-\*.jar
wordcount /lyrics/input /lyrics/output

hdfs dfs -cat /lyrics/output/part-r-00000

The output correctly displayed word frequencies, confirming successful
execution of the Map and Reduce phases.

------------------------------------------------------------------------

## 2️⃣ Map Phase Details

### Input to Mapper

-   Key → LongWritable (Byte offset)
-   Value → Text (Line of text)

Example: (0, "We're up all night till the sun")

### Output from Mapper

-   Key → Text (Word)
-   Value → IntWritable (1)

Example: ("night", 1)

------------------------------------------------------------------------

## 3️⃣ Reduce Phase Details

### Input to Reducer

("up", \[1,1,1,1\])

Type: - Key → Text - Value → Iterable`<IntWritable>`{=html}

### Output from Reducer

("up", 4)

Type: - Key → Text - Value → IntWritable

------------------------------------------------------------------------

## 4️⃣ Custom WordCount Implementation

-   Implemented map() and reduce() functions\
-   Removed punctuation using replaceAll()\
-   Used StringTokenizer for tokenization

Compilation:

javac -classpath `hadoop classpath` WordCount.java\
jar cf wordcount.jar WordCount\*.class

Execution completed without errors.

------------------------------------------------------------------------

## 5️⃣ HDFS Replication Concept

Directories do not have replication because: - Replication applies only
to file blocks\
- Directories store metadata only\
- Metadata is managed by NameNode

------------------------------------------------------------------------

## 6️⃣ Performance Tuning -- split.maxsize

Parameter:

mapreduce.input.fileinputformat.split.maxsize

Controls number of Mapper tasks.

-   Small split → More mappers, more overhead\
-   Large split → Fewer mappers, less parallelism

Optimal performance depends on balancing parallelism and overhead.

------------------------------------------------------------------------

# 🟢 PART 2 -- Apache Spark

Dataset: Project Gutenberg books

Schema:

file_name (string)\
text (string)

------------------------------------------------------------------------

# 📘 Metadata Extraction

Regex Used:

-   Title → (?i)Title:`\s*`{=tex}(.+)\
-   Release Date → (?i)Release Date:`\s*`{=tex}(.+)\
-   Language → (?i)Language:`\s*`{=tex}(.+)\
-   Encoding → (?i)Character set encoding:`\s*`{=tex}(.+)

Extracted using Spark regexp_extract().

------------------------------------------------------------------------

# 📊 Analysis Performed

-   Books released per year\
-   Most common language\
-   Average title length

Challenges: - Missing metadata\
- Inconsistent formatting\
- Noise in text

------------------------------------------------------------------------

# 📘 TF-IDF & Book Similarity

## Term Frequency (TF)

Measures frequency of word in document.

## Inverse Document Frequency (IDF)

IDF = log(N / df)

## TF-IDF

TF-IDF = TF × IDF

Highlights distinguishing words and reduces importance of common words.

------------------------------------------------------------------------

## Cosine Similarity

Cosine(A,B) = (A·B) / (\|\|A\|\| \|\|B\|\|)

-   1 → Highly similar\
-   0 → Not similar

Suitable because it normalizes document length and works well with
sparse vectors.

------------------------------------------------------------------------

# 📘 Author Influence Network

Definition: Author A influences Author B if: 0 \< (year_B - year_A) ≤ X

Represented as Spark DataFrame of directed edges: (author1, author2)

Computed: - In-degree\
- Out-degree

------------------------------------------------------------------------

# 🛠 Technologies Used

-   Apache Hadoop\
-   HDFS\
-   Java\
-   Apache Spark\
-   PySpark\
-   Spark MLlib\
-   Regular Expressions\
-   TF-IDF\
-   Cosine Similarity

------------------------------------------------------------------------

# 📌 Conclusion

This project demonstrates distributed computing fundamentals using
Hadoop and scalable analytics using Apache Spark. It includes MapReduce
implementation, performance tuning, metadata extraction, document
similarity analysis, and influence network construction.

