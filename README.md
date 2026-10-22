# Wu_MA289_SVM_Spoken_Digit_Classification_67

This project uses classical machine learning to distinguish recordings of the spoken digit six from recordings of seven. Each recording is represented by a fixed-length vector of 16,384 numerical audio features. The goal is to estimate the probability that a recording contains the word seven.

The project documents my exploration of support vector machines (SVMs), including feature scaling, principal component analysis (PCA), RBF kernels, and hyperparameter tuning. I used Python and scikit-learn in Google Colab.
Results
Best observed Kaggle public leaderboard ROC AUC: 0.96386. I experimented with changing the param_svm__C, param_svm__gamma, and param_pca__n_components to reach this value. However, I had submitted it the day after the deadline, so it might not show on the leaderboard. In Step 10 of the collab code, you can see the different paramete_grids I was running to find the best values. Then, manually choosing that to create a submission using the main model made and submitting and iterating from there.

Data and labels
- Training data: 400 labeled recordings (200 of each digit)
- Competition test data: 200 unlabeled recordings
- Input features: x0 through x16383
- ID: recording identifier, not a predictive feature
- Original labels: 0 = six; 1 = seven
- SVM training labels: -1 = six; +1 = seven
The submission format has exactly two columns: ID and Probability, where Probability is the estimated probability of class +1 (seven).

Approach
1. Separate predictors (X) from labels (y), excluding the ID column.
2. Recode labels from {0, 1} to {-1, +1} for SVM interpretation.
3. Standardize numeric features with StandardScaler.
4. Reduce feature dimensionality with PCA.
5. Fit an RBF-kernel SVC and examine C, the number of PCA components, and gamma.
6. Compare configurations using stratified cross-validation with ROC AUC as the scoring metric.
7. Retrain the selected model on the labeled dataset and generate probabilities for the held-out competition test set.
The scaler and PCA are inside a scikit-learn Pipeline so each cross-validation fold fits preprocessing only on its training portion, reducing data leakage.

Key concepts I learned
- Margin: SVMs seek a separation boundary with a wide gap between classes.
- Support vectors: Observations closest to or inside the margin help determine the boundary.
- Soft margin / C: In scikit-learn, larger C penalizes margin violations more strongly. A larger value is not always better.
- RBF kernel / gamma: Controls how quickly the similarity of two feature vectors decreases with distance. My leaderboard results were particularly sensitive to gamma.
- PCA: Reduces the number of features; keeping additional components can preserve information but may add noise or complexity.
- ROC AUC: Measures how well a model ranks positive examples above negative examples; it is not the same as classification accuracy.
