# Alzeimer-prediction
Alzheimer prediction using the gene data type

Many diseases have received considerable attention in recent years but AD has not due to shortage of data. I intend to present a systematic framework for prediction AD with minimal gene set with considerable accuracy. This project can be a medical aid in predicting AD through GE.
I worked with GSE33000 gene series in order to predict AD. Our work includes 4 stages. The first step was selecting the right dataset. The dataset was selected after thorough understanding and the availability.
The second stage was preprocessing the dataset. This included transforming the dataset columns and well defining the data to maximize its utility. The gene selection occurred using 3 user defined filter metric applied on all the columns of the dataset. The intersection was then used to predict AD.
The dataset was inspected for missing values and they were handled using mean as the measure.
The third phase included training the ML algorithms to perform the binary classification (AD patients and non- AD patients).
The last phase of the project includes providing the test dataset to the powerful and efficient ML algorithm.
I obtained Logistic Regression to the most effective with the genes et provided. Finally for credible performance we used six metrics to evaluate the performance of our algorithm. We used stratified k fold validation on all these six metrics for validation.
Alzheimer's prediction using MRI is not effective due to the difficulty in sampling
the posterior brains of AD patients. The solution to this is using gene expression to predict
AD.
This project aims at providing an accurate prediction of AD with minimal gene set. Our main focus is to select the best machine learning algorithm to obtain accurate results.
This project is distinguished by its mechanism of gene selection and thus provides a systematic framework for the prediction.
The project follows an approach to perform different gene selection techniques to obtain the intersection of all the three sets.
The obtained values are then compared on two different ML algorithms and are compared against six performance metrics.
The obtained result is used to provide a better understanding of factors responsible for causing AD.

![image](https://github.com/user-attachments/assets/22233e21-abb7-45de-bedd-88fb7b5a244a)


### Explanation of the Code

This script performs feature selection and classification on a dataset using Support Vector Machine (SVM) and Logistic Regression (LR) models. The process begins by loading and preprocessing the dataset, including handling missing values with mean imputation and encoding target labels. Three feature selection methods (chi-squared, ANOVA, and mutual information) are used to select the most relevant features. The selected features are then split into training and test sets. 

A repeated stratified k-fold cross-validation is used to train and evaluate the SVM model, capturing metrics such as accuracy, precision, recall, specificity, F1 score, ROC AUC, and Cohen's kappa. The confusion matrix is plotted to visualize the classification performance. The same process is repeated for the Logistic Regression model. The performance of both models is compared using bar charts that display AUC, accuracy, and precision.

In summary, the code implements a comprehensive pipeline for feature selection, model training, evaluation, and comparison of two machine learning algorithms on a binary classification task.

