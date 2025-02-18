# Bank Marketing Campaign Analysis

This notebook analyzes a dataset from a Portugese banking institution, focusing on the effectiveness of multiple marketing campaigns for term deposits.  The goal is to build and compare classification models to predict customer subscription rates.

# Data Source

The dataset originates from the UCI Machine Learning repository: [https://archive.ics.uci.edu/ml/datasets/bank+marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

Additional information and a detailed description of the features can be found in the accompanying paper: [CRISP-DM-BANK.pdf]

# Problem Definition

The business objective is to develop a model that accurately predicts whether a customer will subscribe to a term deposit based on various customer attributes and marketing campaign data.  This allows for more targeted marketing efforts and improved resource allocation.

# Methodology

1. **Data Exploration:** Analyze the dataset for missing values, data types, and potential feature engineering opportunities.
2. **Feature Engineering:** Encode categorical variables and scale numerical features.
3. **Model Training:** Train several classification models: Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Machines (SVM).  A baseline model is established for comparison.
4. **Model Evaluation:** Evaluate model performance using accuracy and training time.  Compare model performance on a held-out test set.
5. **Model Improvement:** Explore feature engineering (e.g., removing irrelevant features), hyperparameter tuning (using GridSearchCV), and alternative performance metrics (e.g., F1-score) to enhance model accuracy.
6. **Visualization:** Use visualizations to understand data distributions, feature relationships, and model results.

## Python code 
https://github.com/Biju-Seth/Marketing_Campaign/blob/main/marketing_campaign_model.ipynb

## Key Findings

**Initial model comparison**
  Model	                    Train Time	Train Accuracy	Test Accuracy
  Logistic Regression	      0.054253	  0.894052	        0.894513
  KNN	                      0.003485    0.911654	        0.884924
  Decision Tree	            1.322128	  0.998938	        0.839524
  SVM	                      42.155427	  0.898088	        0.895606


**After Hyperparameter tuning**

Model	                      Train Time	    Train Accuracy	  Test Accuracy	  Train F1	    Test F1
Logistic Regression	        7.152557e-07	    0.894052	        0.894513	      0.286532	    0.302008
KNN	                        1.192093e-06	    0.903369	        0.891114	      0.394447	    0.323019
Decision Tree	              9.536743e-07	    0.906191	        0.895242	      0.454754	    0.401940
SVM	                        9.536743e-07	    0.898088	        0.895606	      0.294834	    0.278523

The F1-score is chosen as the primary evaluation metric because it provides a balanced measure of precision and recall. In this banking context, where the goal is to predict customer subscription to a term deposit, both false positives and false negatives have significant implications. A high precision is desirable to minimize marketing costs associated with contacting customers unlikely to subscribe. Meanwhile, a high recall is important to ensure that potential customers who would subscribe are not missed. The F1-score effectively balances these two objectives, providing a more holistic evaluation compared to accuracy alone, especially when dealing with imbalanced classes.

From the above evaluation, Decision Tree model came out with a higher set of accuracy, more importantly with a higher F1 score. Since the F1 score is still not in the higher range, closer to 1, the correlation is still weak among all the models.


# Next Steps and Recommendations:

1. **Feature Engineering:**
   Explore feature interactions. For example, create a combined feature for age and job type to capture potential synergistic effects.
      Consider creating additional categorical features based on thresholds of numerical features.

2. **Advanced Modeling:**
   Explore more sophisticated models like Random Forests, Gradient Boosting, or Neural Networks to potentially improve the F1-score and accuracy.
      Carefully consider the class imbalance in the target variable.  Techniques such as oversampling the minority class or using weighted loss functions can address potential biases towards the majority class and provide a more accurate prediction for subscription.
      Implement cross-validation with more folds.  Current implementation is 5 fold cross validation, consider increasing the number of folds for better generalization.

3. **Evaluation and Refinement:**
   Evaluate the chosen model rigorously on a held-out validation set to assess its true performance in real-world scenarios.  Ensure this held-out set is not used in training or tuning phases.
      Conduct A/B testing to compare the performance of the model-driven targeting strategy with the existing methods to quantify the actual impact of these model driven improvements.

4. **Monitoring:**
   Continuously monitor the model's performance over time as customer behavior and market conditions change.  Retrain and update the model periodically to maintain its accuracy and relevance.

