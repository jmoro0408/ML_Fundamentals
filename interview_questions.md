## Potential Questions


# What is regularization? How does it work?
Regularisation is used to attempt to control overfitting.


There are multiple ways to implement regularisation; some of the most common are:
L1 (Lasso), L2 (Ridge), ElasticNet, Early Stopping, Dropout


L1 - Works by adding a new term to the loss function. This term takes the sum of all the model coefficients. By attempting to minimise these coefficients as the loss is minimised simpler models are preferred and overfitting is reduced.


L2 - Same as L1 however the sum of the squares of the coefficients are added to the loss function instead. This means the actual coefficient values will approach zero but never equal zero.


ElasticNet - Combines L1 and L2 regularisation.


Early Stopping - Algorithmic loss is tracked as the algorithm trains. When the decrease in loss is below some specified threshold (eta) for a number of epochs (patience) the training is stopped.


Dropout - Relevent for neural networks. Randomly sets a portion of the neurons to zero through traning. By dropping random neurons the model is forced to learn from all of it's inputs rather than over-relying on a few. This can help generalization.



# Whats the difference between Lasso (L1), ridge (L2) and elastic net regression?
L1 regression uses the absolute sum of the coeffients in minimising the loss.
L1 Regression can force coefficients to zero
L1 Regression can be useful when you have lots of useless features (as some can be removed completely)
L2 Regression uses the sum of the square of the coeffcients when minising loss
L2 Regression can force coefficients to close to zero but never zero
L2 Regression is useful when you have lots of useful features (as they will never be completely removed)
Elasticnet combines L1 and L2 regression


$$ ElasticNet = Least Squares + Lasso + Ridge$$
$$Loss(Data \vert Model) + \lambda_{2} (|w_1| + \ldots + |w_n|)+ \lambda_{1} (w_1^2 + \ldots + w_n^2) $$



# How would you deal with lots of features? (High cardinality?)
* A categorical feature with lots of unique values can cause issues with certain algorithms, as they may need to be encoded as numeric values. For example, postcodes in the UK are split into 3000 districts with 1.8million unique codes in total. Encoding these as One-hot vectors can cause a massive increase in the number of features, with most being sparse (curse of dimensionality).
To overcome this you can use a tree based algorithm, which can handle categorical features.
- Binning
Alternatvively, you can bin features into different categories, for example combining all scotish postcodes to "Scotland".
You can also pick a threshold value and categorise any count of values below that threshold into "Other". E.g say we have  five colours: Red (50), Blue(40), Yellow (5), Green (3) and Orange (2). We pick a threshold of 90 and keep "Red" and "Blue" (40 + 50 = 90). Everything else gets binned into "other". This reduces the number of features from 5 to 3.
- Types of encoding
Ordinal - assigns an integer to each value. i.e red -> 1, blue ->2, green ->3. However the algorithm learns that green is 3x "bigger" than red, which doesnt make sense for most applications. Can be useful when categorical features really do map to some idea of "bigger", e.g if review results are "awful", "bad", "okay", "good", "great" we can say that "good" is in some ways "bigger/better" than "awful".
Hashing - Mapping each unique value in the category to a integer in a predetermined range. Collision can occur (where different values are assigned the same integer) if not careful. I.e mapping 1000 unique values to 10 integers will certainly result in collision.
Word Embeddings - We can use algorithms such as Word2Vec to encode words into numbers. This uses a pretrained neural network to identify words that are close to each other contextually and assign them a vector representation of numbers. The size of the vector depends on the algorithm used.

# What is overfitting/underfitting? How can you avoid it?
* Overfitting is when the model learns to fit to the noise of the data rather than generalise well to the overall patterns. This is also known as having high variance and low bias. 
* Underfitting is when the model is too simple to learn the intricacies of the data and does not capture the underlying patterns well. This is also know as having low variance and high bias. 
* Avoiding
Overfitting can be avoided by:
1. including reguarlization
2. Monitoring train/validation loss and early stopping
3. increasing the diversity and amount of data
4. Using ensemble methods
Underfitting can be avoided by:
1. Removing/decreasing reguarlization
2. Inreasing the duration of training
3. Feature selection - adding more useful predictive features
4. Choosing a more complex model


# What are false positives/false negatives, what are FPR/TPR?
False positive = predicting a positive outcome when the real outcome is negative
False negative = predicting a negative outcome when the real outcome is positive
the false positive rate is given by:

$$FPR = \frac{FP}{FP+TN} = \frac{FP}{N}\;where\;N\;=\;The\;total\;number\;of\;ground\;truth\;negatives$$ 

The FPR can be used as a metric to determine the performance of classification problems, it is especially useful when the cost of incorectly identifying a positive is high. 

True positive rate is defined as :


$$TPR = \frac{TP}{TP+FN} = \frac{TP}{P}\;where\;P\;=\;The\;total\;number\;of\;ground\;truth\;positives$$

TPR is also known as Recall, see below. 

# What is precision and recall?
Recall can be thought of as the percentage of total positive cases captured correctly.

Precision is the percentage of predicted positives that are actually correct. 

In reality there is a trade off between precision and recall, as one increases the other decreases. 
# How does adjusting the classification threshold affect precision and recall
By default, the threshold for classifying binary targets is 0.5. i.e. any predictions > 0.5 = positive, and <0.5 = negative.
By adjusting the threshold we can tell the classifying to adjust the precision and recall of the outputs. 

Example 1: Adjust threshold to 0.9. 
i.e. predictions with a log likelyhood > 0.9 = positive, <0.9 = negative. 
This means our classifier is not predicting as many positive outcomes, but it is very confident in the ones it does predict. 

E.g we have decreased the number of **↓False positives** and increased the number of **↑False Negatives** therefore our **↑precision** has increased and our **↓Recall** has decreased

Example 2: Adjust threshold to 0.1. 
i.e. predictions with a log likelyhood > 0.1 = positive, <0.1 = negative. 
This means our classifier is predicting a lot of positive outcomes but is not confident in them. 

E.g we have decreased the number of **↓False Negatives** and increased the number of **↑False positives** therefore our ** ↓Recall** has increased and our **↑precision** has decreased. This can be better simplified in a table:

| Threshold | Precision | Recall    | Description                                                                                                    | Possible Scenario                                                                                                    |
|-----------|-----------|-----------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| 0.9       | Increased | Decreased | our classifier is not predicting many positive outcomes, but it is very confident in the ones it does predict. | Cost of misclassifying a positive outcome is high. e.g diagnosing cancer                                             |
| 0.5       | Default   | Default   | Usual case for classification                                                                                  | Cost of identifying TP/TN is balanced                                                                                |
| 0.1       | Decreased | Increased | our classifier is predicting a lot of positive outcomes but is not confident in them.                          | Cost of misclassifying a positive outcome is low, but missing actual positive is high. e.g predicting customer churn |

# What is F1 Score?
F1 score is the harmonic mean of precision and recall. It is useful to compare the performance of different classifiers, as comparing two values for each can be difficult to understand. 

It balances the performance of precision and recall into one number. 
It is particuarly useful when considering the performance of classifiers on imbalanced data. 


# What is bias and variance?
High bias = underfitting
High variance = overfitting
The bias-variance tradeoff describes achieveing a model that generalizes well to new data but is complex enough to capture the patterns and make correct predictions. 

# How can you handle missing data?
1. Imputation - can impute based on mean,median,mode depending on distribution
2. Deletion - Delete rows/features that are missing depending on count
3. Use missingness as a feature - create new representationf or missing i.e set all missing as -1

# What is the difference between a random forest and decision tree?
A random forest is an enemble of decision trees that averages the outputs of all the trees and "votes" on the predicted class

# What is a confusion matrix?
* A way of visualizing the count of predicted True positives/false positives/true negatives and false negatives 
# What metrics would you use for regression, classification?
* Regression - RMSE, MSE, MAE
* Classification - F1 Score, Accuracy, AUC, PRecision/recall, 
# How can you deal with imbalanced data?
* Choose the right metrics - For imbalanced data Accuracy is useless, better to use F1 score and precision and recall
* Udersample - Reduce the count of your majority target examples to match that of your minority - downside is that you lose (potentially a lot of ) data. 
* Oversample - duplicate minority class examples to match that of you majority class. Downside is your model does not gain any varierty in the data and increases the chance of overfitting. 
* SMOTE - Synbthetically creates new data for oversampling, does not duplicate same data. Better than plain oversampling for varience in data but can create data that is non feasable. i.e Postcodes that dont exist or chairs with 5 legs
* Manually set class weights - in some algorithms class weights can be set to tell the model to emphasise specific target classes. 
* USe a tree based model - they are resiliant to imbalanced data

# What is the difference between supervised and unsupervised learning?
* Supervised - Uses label input and output data
Can be divided into classification, regression, 
Trains by making predictions on the data and adjusting the parameters to increase accuracy. 

* unsupervised - Doesnt require labelled date or human input to train
- CLustering, assosciation, dimensionality reduction, 
Clustering - Groups similar groups together. e.g customer segmentation
Assosciation - Assosciates similar items together. i.e "customers who bought this also bought X"
Dimensionality reduction - Reduces the number of features in a dataset through PCA (Principal component analysis,)/autoencoders etc. Can be used to reduce the curse of dimensionality or othen for casting features down to 2D space for plotting
* Dont make predictions - only group data together
* Lack of transparency on how data is clustered i.e PCA features are not clear how they're built up

# What is cross validation?
Cross validation continuously measures the performance of the algorithm throughout training. It does this by splitting the training data into train/test chunks and measuring performnace on the test chunk. Usually this is done *n* times and the metrics averaged 
# What is an activation function?
An ativation function is part of a neural network that input non-linearity into the system. 
Common function are sigmoid, reLU, tanh. 
They are required because if you chain a lot of linear function togehter like in a DNN you still only get  alinear function, so non-linearity is required to capture complex patterns. 
DNNs without activation functions are nonlinear because they are just adjusting the weights and biases. It’s because it doesn’t matter how many hidden layers we attach in the neural network; all layers will behave in the same way because the composition of two linear functions is a linear function itself.

All hidden layers in a DNN will use the same activation function, however the choice of activation function in the output layer will depend on the task. 

For regression you need a continuous function - i.e Relu, 
For binary classification  sigmoid is typically used
for multiclass classification we need softmax as this provides the probabilities over all classes. 

# What is the curse of dimensionality?
* Too many sparse feaures (i.e from the output of one hot encoding caegory with many unique vals) makes it difficult for the algorithm to learn and is computationally expensive
# What are outliers? How would you deal with outliers?
* Outliers are data points that do not fit with the rest of the data. Can be identified through plotting i.e box plots, histograms or just sorting the data and investigating the min/maxes
* Can use z scores to identify how far from the mean a point is (for normally disributed data). 
* Can use interquartile range to fence outliers and detect if outside 75 (or other) %

If the outlier is legitimate it can be left in and may influence the predictions

# What is the typical ML lifecycle?
1. Frame and understand problem
2. Understand business needs
3. Speak to stakeholders
4. Identify type of problem - regression, classification, supervised, unsuprvised
5. understand how performance should be measured
6. understand any comparable problems/manual attempts
7. Get the data
8. Get legal obligations of using the data
9. Create environment
10. Explore data (EDA) (See below)
11. remove outliers (optional)
12. fill in missing values - impute or remove
13. drop useless features
14. feature engineering (see below)
15. Scale/normalize features
16. handle categorical data
17. shortlist models
18. run through bsic models with no tuning to get baseline performance
19. take forward 2/3 models for tuning
20. fine tune, iterating on feature engineering/cleaning/transformation
21. Get final score on test set
22. present solution to stakeholders
23. Deploy solution
24. Upload to cloud service
25. For bath prediction - set airflow/cron to retrain model on new data 
26. for online prediction - setup flask/fast api service for model exposure



# What’s the difference between normalization and standardisation?
Normalization typically means rescales the values into a range of [0,1]. 
Generally Normalization is better when the data is non-gaussian or the distribution is unknown.

Standardization typically means rescales data to have a mean of 0 and a standard deviation of 1 (unit variance).

# Explain KNN
K nearest neighbours is a supervised algorithm used for both regressin and classification, although typically used for classification. The thinking is data that is of the same class is "close" to each other. 
It measures the distance (usually euclidean) from a new data point to the existing data and assigns the label most freuqency seen in the *k* nearest neighbours to this point. 

Choosing *k* is very important and is typically done by plotting the train and test error for various values of k and using the elbow method. 

# Explain K-means clustering
Type of clustering algorith:
1. Choose the number of clusters
2. Randomly initiaize centroids in the data
3. compute the distance from the centroids to all data point using some distance metric
4. assign each data point the class label of the centroid nearest it
5. take the mean of each cluster and set this point to be the new centroid for that cluster
6. repeat the algorithm until the mean centroid doesnt change


Choosing number of clusters is key, this is usually done by eduated guess by inspecting the data and understanding the business problem

Initial location of the centroid can affect the final outcome, usually the algorithm is run a few times to get an average

# Explain PCA
PCA is a form of dimensionality reduction, commonly used to reduce the features in a dataset or to cast down to 2 for 2D plotting. 


PCA combines highly correlated features into a new smaller seet of data called compoennts which preserve most of the data varience. 


# How would you conduct an AB test
AB testing is used when you want to determine if an outcome is just down to chance or is statistically significant. 
1. Devlop hypothesis. 

Null hypothesis is what you are trying to disprove i.e changint he page layout will not drive more clicks. 

Alternate hypothesis is the opposite i.e new page layout will increase clicks. 

2. devlop test statistic

What you are going to measure for performance. i.e click through rate, number of purchases, time take for patient to recover. 

3. choose test and control group 

Randomly sample groups and ensure your sample size is big enough for coverage bias

4. Collect data

5. Calculate p-value 

P-value tells you the probability the outcome of the test is a result of chance. If the probablility is less than some threshold (typically 5%) you can say the outcome is statistically significant and not due to to chance. 

i.e you reject the null hypothesis and accept the alternative hypothesis

p-value can be calculated with statsmodels in python. 

# What are some common biases?
Confirmation bias
1.	Preconceived notions impact how information is interpreted. i.e refs calls against your team. Remembering news stories and articles that align with your beliefs. 

Selection bias

1.	When population samples don’t accurately represent entire target group. 

I.e surveying one neighbourhood of a city is not representative of the entire city
Random sampling and stratified random sampling

Outlier bias

Using mean when median is more appropriate. i.e distribution of wealth in a room.

Omitted variable bias

Correlation vs causation. i.e project yellow -> one person may be correlated with more non compliances, but when included more variables it seems its because he was designing more complex buildings with  a different building code


# How is an ROC curve calculated? When would you use one?
ROC =- receiver operator characteristic curve. 

it is a performance measure for classification problems, commonly used for binary but can also be used for multiclass. 

Commnly used with area under the curve (AOC) to get a single value for the ROC chart. 

It is calculate by varying the prediction threshold and claculating the True positive rate and false positive rate for each point. A chart of the TPR/FPR is then drawn. 

AUC calculates the area under the chart, an AOC of 1 is perfect, AOC of 0.5 is equal to a random classifier. 

ROC suffers when used in imbalanced datasets as the FPR/TPR can be skewed by over calculating one class


# What is backpropogation?
Backpropogation is the method by which neural netowkrs learn. 
1. Each batch of data is passed through the NN (the feedforward stage). Here, the algorithm computes the outputs of all the neurons in eahc layer and passes it to the next, continuing until we get to the output layer

2. The output layer makes predictions based on the incoming data and the activation function. 

3. The algorithm calculats the error using the loss function 

4. It computes how much each output connection contributed to the error (using the chain rule)

5. The algorithm measures the error contribution from each neuron in the network by propogating the error gradient back through the network

6. Gradient descent is performed on all neurons to tweight the weights in the network, based on the error gradients it just calculated 

# What is gradient descent? How does it work?
Gradient descent is a method of optimization. The idea of GD in ML is to tweak the algorithm parameters to minimise the cost function. 
The aklgorithm calculated the derivative of the cost function at a given point and tweaks the parameter in the direcvtion of decreasing cost until a minima is reach. 

Learning rate affects the time taken to converge and chance of convergance 

Stochastic gradient descent is quicker as it calcuates gradients on a random sample of data rather than every point but is not guaranteed to reach the minimum, howevber will come close. 

# What are weights and biases?
Weights and biases are learnable parameters in ML algorithms. 

In linear regression they can be thought of as the gradient (weights) + intercept (bias). 
In NN each neuron is essentially a linear model with a non-linear activation function. 

Weights and biases can be adjusted through gradient descent by measuring the error and adjusting the parameters accordingly to minimise this error, or sometimes by formula for dcertain algorithms. 

# How do LLMs work?
# Whats the difference between boosting and bagging?
# How do transformers work?


# Explain logistic regression/Linear regression/decision tree
# Explain Bayes Theorem


# Potential Qs - Specific Interview
1. Why normalization over scaling? -> better when distribution is non gaussian
* Generally Normalization is better when the data is non-gaussian or the distribution is unknown. Future work would include trying a number of different transformation techniques (power transform, StandardScaler, robustScaler) on features indibvidually and checking the performance.



2. Do I need to worry about CV splitting due to time component?
* For time series data, yuo generally need to ensure both the CV split and train/test split are setup correctly. As the "future" data is often directly impacted by the "younger" data, the split needs to ensure any patterns generated by the younger data is captured. This is not typically done in a CV/trainTestSplit as the splits are made randomly.
SKLearn's TimeSeriesSplit can be used to split the data into "rolling" splits, ensuring the data is continuouysly split based on time.


![Alt text](image.png)


An example is the stock market - where todays stock price is directly influence by yesterdays/last weeks.



3. Why does seting class weights make it even worse?
* Not sure. Experimenting with heavily imbalanced data shows that the LogReg "Class weights" argument doesnt seem to help at all, either when set manually or with 'balanced' param.


4. What do SHAP waterfall plots actually represent?
The x-axis represents the log-odd usints of the XGBoost classifier output. That means that negative values imply probabilities of less than 0.5 -> negative prediction.
Future work - use SHAP waterfall and force plots to examine where the classifier made incoorrect predictions.


5. Why F1 over AUC?


AUC is more sensitive to class imbalance.
Reminder: TPR = TP/(TP+FN)
            FPR = FP/ (FP + TN)



ROC curve is not a good visual illustration for highly imbalanced data, because the False Positive Rate ( False Positives / Total Real Negatives ) does not drop drastically when the Total Real Negatives is huge.


AUC was attempted throughout the modelling, however this effect was seen first hand as scores of >0.95 were seen when the classifier made predictions of all 0.


# What is EDA? What are some techniques commonly used throughout EDA?
# How can  you increase the number of features in a dataset?
# Why does ML need a lot of data? What happens when we dont have enough?
# How does a model learn during training?
# What are ensemble methods?