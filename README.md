Confidence-Based Abstention in Recommender Systems (Hybrid: Collaborative + Content-Based)

1. Overview

This project implements a hybrid recommendation system that combines:

 Collaborative Filtering (SVD)
 Content-Based Filtering (TF-IDF + Cosine Similarity)
 Confidence-based filtering & cold-start handling**

The system is built using the Amazon Clothing, Shoes & Jewelry Reviews dataset and demonstrates end-to-end recommendation pipeline development — from data preprocessing to evaluation.


2. Features

 Data preprocessing and filtering (removes sparse users/items)
 Exploratory data analysis with visualizations
 Collaborative filtering using **SVD (Surprise library)**
 Content-based filtering using **TF-IDF**
 Hybrid recommendation model (weighted scoring)
 Confidence-aware recommendations (abstention mechanism)
 Cold-start handling for new users
 Evaluation metrics:

  * Precision@K
  * Recall@K
  * NDCG@K
  * MAP@K



3. Tech Stack

* Python
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* scikit-surprise



4. Dataset

* Source: Amazon Reviews Dataset
* Category: Clothing, Shoes, and Jewelry
* File: `reviews_Clothing_Shoes_and_Jewelry_5.json`

The dataset is downloaded and processed directly in the script.



5. Installation

```bash
pip install scikit-surprise
pip install pandas numpy==1.26.4 matplotlib seaborn scikit-learn
```

---

6. How It Works

> Data Processing

* Loads JSON review data
* Extracts:

  * user_id
  * item_id
  * rating
  * review text
* Filters users/items with fewer than 5 interactions



> Collaborative Filtering (CF)

* Uses **Singular Value Decomposition (SVD)**
* Learns latent user-item interactions
* Predicts ratings for unseen items

```python
recommend_cf(user_id, n=10)
```


> Content-Based Filtering

* Uses **TF-IDF vectorization** on review text
* Computes **cosine similarity** between items

```python
recommend_content(item_id, top_n=10)
```

> Hybrid Recommendation System

Combines:

* CF score (60%)
* Content similarity (30%)
* Confidence score (10%)

```python
hybrid_recommend(user_id, top_n=10)
```


> Confidence-Based Recommendations

* Avoids uncertain predictions
* Filters recommendations using a confidence threshold

```python
recommend_with_abstention(user_id, threshold=0.4)
```



> Cold Start Handling

* New users → Popular items
* Existing users → Hybrid of content similarity + popularity

```python
cold_start_user_recommend(user_reviews, top_n=10)
```



7. Evaluation Metrics

The model is evaluated using:

| Metric      | Description                |
| ----------- | -------------------------- |
| Precision@K | Relevant items in top-K    |
| Recall@K    | Coverage of relevant items |
| NDCG@K      | Ranking quality            |
| MAP@K       | Mean average precision     |



8. Visualizations

* Rating distribution
* User-item heatmap
* Hybrid score components
* Evaluation metric comparison



9. Example Usage

```python
user_id = df['user_id'].iloc[0]

print("CF:", recommend_cf(user_id))
print("Hybrid:", hybrid_recommend(user_id))
```


10. Future Improvements

* Integrate **Deep Learning models**:

  * Variational Autoencoders (VAEs)
  * Generative Adversarial Networks (GANs)
* Improve scalability for large datasets
* Add real-time recommendation API
* Deploy using Flask or FastAPI


11. Authors

 Devansh Chauhan


12. License

This project is for academic and educational purposes.



13. Acknowledgements

* Amazon Review Dataset (Stanford SNAP)
* Surprise Library for recommendation algorithms
* Scikit-learn for feature engineering



14. Summary

This project demonstrates how combining **collaborative filtering + content-based methods + confidence scoring** leads to more **robust, accurate, and practical recommendation systems**, especially when handling **sparsity and cold-start problems**.
