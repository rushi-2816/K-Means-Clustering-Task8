# Task 8: Clustering with K-Means

## 🔍 Objective
Perform unsupervised learning using the K-Means algorithm to segment customers based on annual income and spending score.

## 📊 Dataset
- **Name**: Mall Customer Segmentation
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **File Used**: Mall_Customers.csv

## 🛠️ Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## ✅ Steps Followed
1. Uploaded and loaded the dataset.
2. Selected features: `Annual Income (k$)` and `Spending Score (1-100)`.
3. Scaled the data using StandardScaler.
4. Used the **Elbow Method** to determine the optimal number of clusters (K).
5. Applied **K-Means clustering**.
6. Visualized the resulting clusters using scatter plot.
7. Evaluated the clustering using **Silhouette Score**.

## 📈 Results
- **Optimal number of clusters (K)**: 5
- **Silhouette Score**: ~0.55 (depends on random initialization)

## 📂 Files Included
- `kmeans_clustering.ipynb` – Jupyter/Colab notebook with full step-by-step code
- `Mall_Customers.csv` – Dataset file
- `README.md` – This file

---

## 💬 Interview Questions & Answers

1. **How does K-Means clustering work?**  
   K-Means partitions data into K clusters by minimizing the variance within each cluster. It does this by iteratively assigning points to the nearest cluster center and updating the centers.

2. **What is the Elbow method?**  
   It's a technique to find the optimal K by plotting the inertia (within-cluster sum of squares) and identifying the "elbow" point where adding more clusters doesn't significantly reduce inertia.

3. **What are the limitations of K-Means?**  
   - Sensitive to outliers  
   - Requires predefined number of clusters  
   - Assumes spherical clusters of similar sizes

4. **How does initialization affect results?**  
   Initialization determines starting cluster centers. Poor initialization can lead to poor results. Using `k-means++` helps in better and more stable outcomes.

5. **What is inertia in K-Means?**  
   Inertia is the total sum of squared distances between each point and its cluster center. Lower inertia indicates tighter clusters.

6. **What is Silhouette Score?**  
   A metric that evaluates how well a data point fits into its assigned cluster vs. others. Values close to 1 are better.

7. **How do you choose the right number of clusters?**  
   Use methods like the **Elbow Method**, **Silhouette Score**, or domain knowledge.

8. **What’s the difference between clustering and classification?**  
   - **Clustering**: Unsupervised, groups data without labels.  
   - **Classification**: Supervised, assigns predefined labels to data.
