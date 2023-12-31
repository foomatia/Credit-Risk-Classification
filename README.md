# Credit-Risk-Classification



## Purpose of the analysis:
The purpose of this analysis is to use various techniques to train and evaluate a model based on loan risk. I am using historical data on lending activity from a peer-to-peer lendince services company to build a model that can identify the creditworthiness of borrowers.

We work with two values,ie. 0 in the “loan_status” column which means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting. The dataset came already with a loan_status column which had to be removed as these values needed to be predicted again. A value_counts() check showed that there are 75036 values for 0 and 2500 values for 1.

There are two parts to the analysis. 
For the first part, the data was split using the train_test_split with a random_state=1. 
Using the original data, a logistic regression model was fit. A prediction was made using the testing data. The model's performance was evaluated using balanced_accuracy_score and a confusion matrix. Finally a report was printed which showed the precision, recall and f1-score for the 0 and 1 labels. 
In the second part, the data was resampled using the RandomOverSampler from the imbalanced library. A value.counts() confirmed an equal split of the data for 0 and 1 to 56277 datapoints. Just like with the original data, a LogistigRegression model was instantiated and evaluated using the same parameters. 

## Results
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.
Following is a summary of the results of both the datasets that were looked at.

### Machine Learning Model 1: Original Data

![Screenshot 2023-12-21 at 13 49 45](https://github.com/foomatia/Credit-Risk-Classification/assets/108195931/9c7218a1-e45d-4090-912d-e97c6e73e65a)

### Machine Learning Model 2: Oversampled Data

![Screenshot 2023-12-21 at 13 50 14](https://github.com/foomatia/Credit-Risk-Classification/assets/108195931/6357f8ae-c86f-408a-a3ec-3f358d21c0fa)

## Summary

Based on the balanced_accuracy_score, Machine Learning Model 2 with the oversampled data seems to perform best. This conclusion is based on the higher score of 0.9942 compared to the score of 0.9514 for Model 1. The score takes into account both precision and recall for each label, provising a balanced measure of overall model performance. 

Looking at the precision and recall data, Model 1 using original data gives a high precision for lable 0 (health loan) but lower recall for lable 1 (high-risk loan). This model may be suitable if priority is to minimize false positives (misclassifying healthy loans as high-risk).
For Model 2 using oversampled data, prevision and recall are high for both labels. This model is suitable if a balanced performance across both labels is important or when misclassifications for high-risk and health loan need to be minimized.

Seeing that both models performed quite well in their own way, I would refrain from recommending either model without further background on the business requirements that iniated this analysis. Specific preference on a higher balanced_accuracy_score, minimizing false positives or chosing which prediction is more relevant (0s or 1s) would play a key difference to chosing which model to pick.  

