# Assignment 2 – Min-Hashing and LSH

**Name:** Arpita Kundu\
**Roll No:** M25DE1004

#### All implementations were developed and executed in Google Colab (Python environment).
------------------------------------------------------------------------

## 📌 Project Overview

This project implements and evaluates:
  -  K-Gram Generation
  -  Exact Jaccard Similarity
  -  Min-Hashing
  -  Locality Sensitive Hashing (LSH)
  -  MinHash & LSH on MovieLens 100k Dataset

The focus is on understanding probabilistic similarity estimation, performance trade-offs, and large-scale similarity search.

------------------------------------------------------------------------
#### 🧩 Q - 1 -- K-Grams Construction
------------------------------------------------------------------------
#### 🧩 Solutions Implemented -

For documents D1.txt, D2.txt, D3.txt, and D4.txt, the following were constructed:
- Character 2-grams
- Character 3-grams
- Word 2-grams
Duplicates were removed (unique k-grams only).

#### Part A: 

Using 3-grams, MinHash signatures were built for D1 and D2 with:
- t = 20
- t = 60
- t = 150
- t = 300
- t = 600
Approximate Jaccard similarities were reported for each value.<br>
<img width="335" height="371" alt="image" src="https://github.com/user-attachments/assets/6034530b-d89c-4e1d-bac5-9d06904cadd4" /><br>

#### Part B:

Exact Jaccard similarity was computed for:
- All document pairs
- All three types of k-grams
Total results reported: 18 similarity values.<br>

<img width="272" height="172" alt="image" src="https://github.com/user-attachments/assets/cd73f739-76f5-4fbb-83b9-560815e43d4b" /><br>

------------------------------------------------------------------------
#### 🔢 Q2 – Min-Hashing
------------------------------------------------------------------------
#### 🧩 Solutions Implemented -
A hash family was constructed with:
- h : k-grams → [m]
- m > 10,000

Part A:

MinHash signatures were generated for:
- t = 20, 60, 150, 300, 600
Approximate similarities were compared.<br>

<img width="302" height="65" alt="image" src="https://github.com/user-attachments/assets/4302bb90-9b9a-4a16-b742-c3bf5f5f569b" /><br>


Part B :

– Best Value of t
- Small t → Faster but less accurate
- Large t → More accurate but slower

Conclusion:<br>
We can conclude to a point that if the value of t is small, the algorithm runs faster but produces lower accuracy in similarity estimation. On the other hand, increasing t improves accuracy because more hash functions reduce estimation error, but it also increases computation time. 
t = 300 provides a good trade-off between computational efficiency and estimation accuracy.

------------------------------------------------------------------------
#### 📈 Q3 – Locality Sensitive Hashing (LSH)
------------------------------------------------------------------------
#### 🧩 Solutions Implemented -

Using:
- t = 160 hash functions
- Similarity threshold τ = 0.7

Part A:

Optimal (r, b) values were selected to create a well-separated S-curve.<br>

<img width="341" height="87" alt="image" src="https://github.com/user-attachments/assets/c64d440f-c475-48a2-9054-2f85de878662" /><br>

<img width="410" height="296" alt="image" src="https://github.com/user-attachments/assets/934c6a42-e6e8-45e3-89ba-0af881a9ac63" /><br>

Part B:

For each document pair, the probability of similarity > τ was computed.<br>

<img width="527" height="301" alt="image" src="https://github.com/user-attachments/assets/db55b795-c6e9-4e8b-83fb-bced6641ffaf" /><br>

------------------------------------------------------------------------
#### 🎬 Q4 – MinHash on MovieLens 
------------------------------------------------------------------------
#### 🧩 Solutions Implemented -
Dataset:
- 943 users
- 1682 movies
- Only movie sets considered (ratings ignored)

#### Steps Performed:
- Computed exact Jaccard similarity for all user pairs
- Selected user pairs with similarity ≥ 0.5
- Computed MinHash signatures using:
      1. 50 hash functions
      2. 100 hash functions
      3. 200 hash functions
- Reported:
      1. Estimated similar pairs
      2. False Positives
      3. False Negatives
      4. Averages over 5 runs<br>
      
<img width="641" height="764" alt="image" src="https://github.com/user-attachments/assets/c327f4ea-35bc-47ed-a7e7-1490fb3bd853" /><br>

<img width="563" height="240" alt="image" src="https://github.com/user-attachments/assets/adb18e69-03e4-463d-ac5d-355b0ceff432" /><br>

<img width="359" height="60" alt="image" src="https://github.com/user-attachments/assets/b50ab12b-f16c-44fe-95ee-af5792618ba6" /><br>

<img width="313" height="54" alt="image" src="https://github.com/user-attachments/assets/374f154a-5666-48d6-9a63-5c9bab7ad663" /><br>

<img width="333" height="60" alt="image" src="https://github.com/user-attachments/assets/3eeae5fc-04b3-4bfd-af72-d7d44422f461" /><br>


------------------------------------------------------------------------
#### 🚀 Q5 – LSH on MovieLens 
------------------------------------------------------------------------

LSH configurations tested:
Hash Functions  	r   	b
50	              5	    10
100	              5	    20
200	              5	    40
200	              10	  20

Similarity thresholds tested:
- τ = 0.6
- τ = 0.8

Observations:
- Increasing bands (b) → fewer false negatives but more false positives
- Increasing rows (r) → fewer false positives but more false negatives
- Higher τ → increases precision but reduces recall

Parameter selection depends on application requirements (precision vs recall trade-off).<br>

<img width="557" height="539" alt="image" src="https://github.com/user-attachments/assets/643e5cee-2180-46db-92ec-4fd6e136239f" /><br>

------------------------------------------------------------------------
#### 🛠 Technologies Used
------------------------------------------------------------------------
- Python 3
- NumPy
- Matplotlib
- Google Colab
- MovieLens 100k Dataset
- minhash dataset

------------------------------------------------------------------------
####  📂 Repository Structure
------------------------------------------------------------------------

<img width="191" height="88" alt="image" src="https://github.com/user-attachments/assets/2015cd93-f96e-469f-a154-808bbd325037" /><br>

------------------------------------------------------------------------
####   📊 Key Learnings
------------------------------------------------------------------------

- Understanding of probabilistic similarity estimation
- Trade-offs between accuracy and computation time
- Effect of banding in LSH
- Evaluation using false positives and false negatives
- Practical implementation on real-world dataset (MovieLens)

------------------------------------------------------------------------
####  🔗 Dataset Reference
------------------------------------------------------------------------
- minhash dataset
    Attached in github repo (Assignment2)
- MovieLens 100k Dataset
    http://www.grouplens.org/node/73

------------------------------------------------------------------------
####  📌 Conclusion
------------------------------------------------------------------------
This project demonstrates practical implementation of:
- MinHash for approximate similarity
- LSH for scalable similarity search
- Parameter tuning for balancing precision and recall
- Experimental validation on real-world datasets

It highlights the trade-off between accuracy, computational efficiency, and scalability in Big Data similarity problems.


Submitted By,

M25DE1004(Arpita Kundu) | Indian Institute Of Technology Jodhpur(IITJ)


