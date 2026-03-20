
## 🎬 (ML with Big Data) Recommender Systems Assignment

📌 Project Overview

This project implements multiple recommendation system techniques using the MovieLens dataset, executed on Google Colab with Google Drive integration.
+ Content-Based Filtering (TF-IDF & User Profiles)
+ Collaborative Filtering (User-Based & Item-Based)
+ Matrix Factorization (SVD)
+ Hybrid Models
+ Neural Network-Based Recommender
+ Reinforcement Learning-Based Recommender
+ Explainability Techniques

📂 Dataset

We use the MovieLens Dataset, which includes:
+ User ratings
+ Movie metadata (genres)
+ User interactions

please refer the [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) dataset (~1MB) on first run. Requires internet access. The dataset is saved to `ml-latest-small/` in the working directory.

⚙️ Installation & Setup<br>

This project is designed to run on Google Colab, with data loaded from Google Drive.<br>

🔹 Step 1: Upload Dataset to Google Drive:

    1. Download MovieLens dataset
    2. Upload it to your Google Drive (e.g., inside a folder like /MovieLens/)
    
🔹 Step 2: Open Notebook in Colab

    Upload or open the notebook:
    1. m25de1004_CSL7110_Assignment3.ipynb
    
🔹 Step 3: Mount Google Drive

    Run this in the first cell:
    1. from google.colab import drive
    2. drive.mount('/content/drive')
    
🔹 Step 4: Set Dataset Path

    Update dataset path in your notebook:
    1. data_path = "/content/drive/MyDrive/MovieLens/"
    2. Make sure it matches your actual folder.
    
🔹 Step 5: Install Required Libraries
    
    In Colab, run:
    1. !pip install scikit-surprise
    2. import pandas as pd
    3. import numpy as np
    4. import sklearn
    5. import matplotlib

🚀 How to Run :

    1. Open the notebook in Google Colab
    2. Mount Google Drive
    3. Update dataset path
    4. Run all cells sequentially

🧠 Implemented Techniques :

🔹 1. Content-Based Filtering

    + TF-IDF on movie genres
    + Cosine similarity
    + Movie-to-movie recommendations
    
🔹 2. User Profile-Based Recommendation

    + Weighted average of TF-IDF vectors
    + Personalized recommendations
    
🔹 3. Collaborative Filtering

    + User-Based CF (similar users)
    + Item-Based CF (similar items)
🔹 4. Matrix Factorization (SVD)

    + Decomposes user-item matrix
    + Predicts missing ratings
    
🔹 5. Hybrid Model<br>

🔹 6. Neural Network Model

    + Learns user & movie embeddings
    + Predicts ratings
    
🔹 7. Reinforcement Learning

    + Multi-Armed Bandit (ε-Greedy, UCB)
    + Q-Learning approach
    
🔹 8. Explainability

    + Feature-based explanations
    + Similar user/item explanations
    + SHAP / LIME concepts

📊 Evaluation Metrics :

    + RMSE (Root Mean Squared Error)
    + Precision@K
    + Recall@K

📁 Project Structure<br>

|--- m25de1004_CSL7110_Assignment3.ipynb<br>
|--- README.md<br>
|--- M25DE1004_CSL7110_Assignment3.pdf<br>
|--- CSL7110 Assignment 3- ML with Big Data.pdf<br>
|--- Datasets<br>


### Notebooks Included :

| Tasks |
|-------|
| Part 1: Content-Based Filtering: |
| Task 1: Implementing TF-IDF Based Recommendation |
| Task 2: User-Profile-Based Content Recommender |
| Part 2: Collaborative Filtering: |
| Task 3: User-Based Collaborative Filtering |
| Task 4: Item-Based Collaborative Filtering |
| Part 3: Matrix Factorization for Recommender Systems: |
| Task 5: Implementing SVD for Recommendations |
| Task 6: Implementing Matrix Factorization with the Surprise Library |
| Part 4: Hybrid Recommendation Model: |
| Task 7: Implementing a Hybrid Recommendation Model |
| Part 5: Learning-Based Recommender Systems |
| Task 8: Content-Based Filtering with a neural network |
| Task 9: Reinforcement Learning in Recommender Systems |
| Part 6: Explainability in Recommender Systems |
| Task 10: Feature-Based Explanations (For Content-Based Filtering) |
| Task 11: Neighborhood-Based Explanations (For Collaborative Filtering) |
| Task 12: Model-Agnostic Explainability (For Deep Learning Models) |
| Task 13: Evaluating Explainability |


⚠️ Notes
+ Mount Google Drive before running
+ Update dataset path correctly
+ Run cells sequentially
+ Some models(SVD, NN, RL) may take longer

---

👩‍💻 Author
Arpita Kundu | IITJ

This assignment is for academic purposes only.

