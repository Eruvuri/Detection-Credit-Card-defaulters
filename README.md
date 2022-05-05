# Detection-Credit-Card-defaulters
This is the final project for the Artificial Intelligence course, CSCI-6600 -02 (Spring-2022), by Eswar Reddy Eruvuri (00754001) and Manish Chandra Singarapu (00744507).

This project involves training a data set with various machine learning techniques and assessing the effectiveness of each model to identify credit card defaulters.

## Table of Contents:
+ [Data Set](#Data_Set) </br>
+ [Machine Learning models used](#Machine_Learning_models_used) </br>
+ [How we used these models in project](#How_we_used_these_models_in_project) </br>
+ [Performance](#Performance) </br>

## <a name="Data_Set"></a> Data Set 

From April to September 2005, this dataset comprises information on default payments, demographic variables, credit data, payment history, and bill statements of credit card clients.

There are 24 features to choose from:

**Demographic Information**
- ID: ID of each client
- LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family / supplementary credit)
- SEX: Gender (1=male, 2=female)
- EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
- MARRIAGE: Marital status (1=married, 2=single, 3=others)
- AGE: Age in years

**Repayment Status for past 6 months**

**(-1 = pay duly, 1 = payment delay for one month, 2 = payment delay for two months, ..., 9 = payment delay for nine months and above)**
- PAY_0: Repayment status in September, 2005 
- PAY_2: Repayment status in August, 2005 (scale same as above)
- PAY_3: Repayment status in July, 2005 (scale same as above)
- PAY_4: Repayment status in June, 2005 (scale same as above) 
- PAY_5: Repayment status in May, 2005 (scale same as above)
- PAY_6: Repayment status in April, 2005 (scale same as above)

**Amount of bill statement for past 6 months**
- BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
- BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
- BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
- BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
- BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
- BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)

**Amount of previous payment for past 6 months**
- PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
- PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
- PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
- PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
- PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
- PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)

**Target Variable**
- default payment next month: Default payment (1=yes, 0=no)
***

## <a name="Machine_Learning_models_used"> </a> Machine Learning models used 
**SVM**

A supervised machine learning approach known as a support vector machine (SVM) can be utilized for both classification and regression tasks. It is, however, mostly employed to solve categorization difficulties. Each of the values at a certain coordinate in the SVM algorithm. Create a point in n-dimensional space (n is the number of features) for each data piece that represents the value of the feature. After that, locate a superplane that clearly differentiates the two classes and classify them.

**How it works**

Drawing a straight line between two classes is how a simple linear SVM classifier works. To put it another way, all data points on one side of the line represent categories, whereas data points on the other side of the line are assigned to other categories. This means you have an infinite number of lines to pick from. The linear SVM algorithm is superior to other algorithms such as closest neighbor in that it selects the best line for classifying data points. Choose a line that separates the data and keep it as far away from the cabinet data points as possible. All machine learning jargon will be easier to understand if you use 2D examples. There are frequently several data points in a grid. You want to categorize these data points, but you don't want them to be categorized incorrectly. This means we're looking for the line that connects the two closest points and separates the other data points. As a result, the two closest data points provide the reference vector for locating this line. The decision boundary is the name given to this line.

**Logistic Regression**

Logistic regression, often called logit regression, binary logit, or binary logistic regression, is a type of statistical analysis. Logistic regression is a statistical approach for predicting the outcome of a dependent variable based on past data. It's a sort of regression analysis that's typically used to tackle difficulties involving binary categorization.

Logistic regression employs a method of modeling the log of the outcome's odds ratio. The coefficients of a logistic regression model are estimated via maximum likelihood estimation. This is because, unlike linear regression, there is no closed-form solution. If you're not familiar with regression analysis, it's a form of predictive modeling approach that's used to figure out how a dependent variable and one or more independent variables are related.

**How it works**

Instead of fitting a regression line, we fit a "S" shaped logistic function in logistic regression, which predicts two maximum values (0 or 1).
The logistic function's curve reflects the probability of things like whether the cells are cancerous or not, whether a mouse is obese or not based on its weight, and so on.
Because it can generate probabilities and classify new data using both continuous and discrete datasets, logistic regression is a key machine learning approach.
Logistic regression may be used to categorize observations based on many forms of data and can quickly identify the most useful factors for classification.

**Naive bayes**

A probabilistic machine learning model called a Naive Bayes classifier is utilized to perform classification tasks. The Bayes theorem lies at the heart of the classifier.

**How it works**

We can calculate the likelihood of A occurring if B has already occurred using Bayes' theorem. The evidence is B, and the hypothesis is A. The predictors/features are assumed to be independent in this case. That is, the presence of one attribute has no bearing on the other. As a result, it is said to as naïve.

**Random Forest**

Random forest is a learning algorithm that is supervised. It creates a "forest" out of an ensemble of decision trees, which are commonly trained using the "bagging" method. The bagging method's basic premise is that combining several learning models improves the overall output.

**How it works**

Random forest is a learning algorithm that is supervised. It creates a "forest" out of an ensemble of decision trees, which are commonly trained using the "bagging" method. The bagging method's basic premise is that combining several learning models improves the overall output.

Random forest has the advantage of being able to solve classification and regression issues, which make up the majority of contemporary machine learning systems. Because classification is frequently regarded the building block of machine learning, let's look into random forest in classification. 

The hyperparameters of a random forest are quite similar to those of a decision tree or a bagging classifier. Fortunately, you may utilize the classifier-class of random forest instead of combining a decision tree with a bagging classifier. You can use the algorithm's regressor to cope with regression tasks with random forest.

While growing the trees, the random forest adds more randomness to the model. When splitting a node, it looks for the best feature from a random subset of features rather than the most essential feature. As a result, there is a lot of variety, which leads to a better model.

As a result, in random forest, the technique for splitting a node only considers a random subset of the features. Instead of searching for the greatest possible thresholds, you may make trees even more random by employing random thresholds for each feature (like a normal decision tree does).

 ## <a name="How_we_used_these_models_in_project"> </a> How we used these models in project
 
 - Performed training on data set.
- Predicting the model by using test data set.
- Evaluated the model by using F1 score,K Fold Cross validation and confusion matrix.
- Testing the trained model with new dataset

## <a name="Performance"> </a> Performance
**Following is the performance of each models used.**
![image](https://user-images.githubusercontent.com/95928967/145607620-8f56e6bf-5f14-4886-a5ff-6506d5abd7b8.png)

![image](https://user-images.githubusercontent.com/95928967/145636792-b7e5e418-02c6-4583-999e-9d0701a92992.png)
