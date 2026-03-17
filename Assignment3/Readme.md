
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

⚙️ Installation & Setup

This project is designed to run on Google Colab, with data loaded from Google Drive.

🔹 Step 1: Upload Dataset to Google Drive:
    1. Download MovieLens dataset
    2. Upload it to your Google Drive (e.g., inside a folder like /MovieLens/)
    
🔹 Step 2: Open Notebook in Colab
    Upload or open the notebook:
    m25de1004_CSL7110_Assignment3.ipynb
    
🔹 Step 3: Mount Google Drive
    Run this in the first cell:
    from google.colab import drive
    drive.mount('/content/drive')
    
🔹 Step 4: Set Dataset Path
    Update dataset path in your notebook:
    data_path = "/content/drive/MyDrive/MovieLens/"
    Make sure it matches your actual folder.
    
🔹 Step 5: Install Required Libraries
    In Colab, run:
    !pip install scikit-surprise

  🚀 How to Run
      1. Open the notebook in Google Colab
      2. Mount Google Drive
      3. Update dataset path
      4. Run all cells sequentially

🧠 Implemented Techniques
🔹 1. Content-Based Filtering

TF-IDF on movie genres

Cosine similarity

Movie-to-movie recommendations

🔹 2. User Profile-Based Recommendation

Weighted average of TF-IDF vectors

Personalized recommendations

🔹 3. Collaborative Filtering

User-Based CF (similar users)

Item-Based CF (similar items)

🔹 4. Matrix Factorization (SVD)

Decomposes user-item matrix

Predicts missing ratings

🔹 5. Hybrid Model

🔹 6. Neural Network Model

Learns user & movie embeddings

Predicts ratings

🔹 7. Reinforcement Learning

Multi-Armed Bandit (ε-Greedy, UCB)

Q-Learning approach

🔹 8. Explainability

Feature-based explanations

Similar user/item explanations

SHAP / LIME concepts

📊 Evaluation Metrics

RMSE (Root Mean Squared Error)

Precision@K

Recall@K

📁 Project Structure
├── m25de1004_CSL7110_Assignment3.ipynb
├── README.md
|-- 
## Notebooks Included

| File | Tasks |
|------|-------|
| `task8_neural_cbf.ipynb` | Task 8: Content-Based Filtering with Neural Network |
| `task9_rl_recommender.ipynb` | Task 9: Reinforcement Learning Recommender |
| `part6_explainability.ipynb` | Tasks 10–13: Explainability (SHAP, k-NN, LIME, Evaluation) |

---

## Dependencies

Install all required packages:

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow shap lime
```

Or with conda:

```bash
conda install numpy pandas scikit-learn matplotlib tensorflow
pip install shap lime
```

**Python version:** 3.8 – 3.11 recommended (TensorFlow requirement)

---

## Dataset

All notebooks **automatically download** the [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) dataset (~1MB) on first run. Requires internet access. The dataset is saved to `ml-latest-small/` in the working directory.

---

## Running Order

1. `task8_neural_cbf.ipynb` — runs independently
2. `task9_rl_recommender.ipynb` — runs independently  
3. `part6_explainability.ipynb` — runs independently (re-loads data internally)

Each notebook is self-contained and re-downloads/re-processes the dataset.

---

## Expected Outputs

| Notebook | Generated Files |
|----------|----------------|
| Task 8 | `task8_training_curves.png` |
| Task 9 | `task9_bandit_comparison.png`, `task9_qlearning_rewards.png`, `task9_comparison.png` |
| Part 6 | `task10_shap_feature_importance.png`, `task12_lime_explanation.png`, `task13_genre_bias.png` |

---

## Notes

- **Task 8 (Neural Network):** Training uses `EarlyStopping` (patience=5) so epochs may vary. GPU is not required but will speed up training.
- **Task 9 (RL):** The simulation uses a subset of 200 users × 200 movies for tractability.
- **Part 6 (SHAP/LIME):** `shap` and `lime` are auto-installed via pip if not present.
