## Metrics Evaluation Lab

Throughout your early career as a Data Scientist you've built article summaries, 
explored NBA talent, and analyzed text on climate change news but you've suddenly realized you need to enhance your ability to assess the models you are building. 
As the most important part about understanding any machine learning model 
(or any model, really) is understanding it's weakness and/or vulnerabilities. 

In doing so you've decided to practice on datasets that are of interest to you, 
and use a approach to which you are (becoming) familiar, kNN. 

Part 1. Select either as a lab group or individuals a dataset that is of interest to you/group. Define a question that can be answered using classification, specifically kNN, for the dataset.

Part 2. In consideration of all the metrics we discussed select a few (threeish) key metrics that should be tracked given the question you are working to solve. 

Part 3. Build a kNN model and evaluate the model using using the metrics discussed in class (Accuracy, TPR, FPR, F1, **Kappa, LogLoss and ROC/AUC**). Make sure to calculate the baserate or prevalence to provide a reference for some of these measures. Even though you are generating many of the metrics we discussed summarize the output of the key metrics you established in part 2. 

Part 4. Consider where miss-classification errors are occurring, is there a pattern? If so discuss this pattern and why you think this is the case. *Look at confusion matrix*

Part 5. Based on your exploration in Part 3/4, change the threshold using the function provided (in the in-class example), what differences do you see in the evaluation metrics? Speak specifically to the metrics that are best suited to address the question you are trying to answer. 

Part 6. Summarize your findings speaking through your question, what does the evaluation outputs mean when answering the question? Also, make recommendations on improvements. 

Recommendations for improvement might include gathering more data, adjusting the threshold, adding new features, changing your questions or maybe that it's working fine at the current level and nothing should be done. 

Submit a .Rmd file along with the data used or access to the data sources to the Collab site. You can work together with your groups but submit individually. 

Keys to Success: 
* Thoughtfully creating a question that aligns with your dataset
* Using the evaluation metrics correctly - some require continuous probability outcomes (LogLoss) while others require binary predictions (pretty much all the rest).
* Evaluation is not about the metrics per say, but what they mean, speaking through your question in light of the evaluation metrics is the primary objective of this lab. 

Notes:
Train and test, 1 data partition
Iris dataset
Select 3 eval metrics that make sense and explain why
Multiclass problem? 


```{r}

#For this example we are going to use the IRIS dataset in R
str(iris)
#first we want to scale the data so KNN will operate correctly
scalediris <- as.data.frame(scale(iris[1:4], center = TRUE, scale = TRUE)) 


str(scalediris)

set.seed(1000)
#We also need to create test and train data sets, we will do this slightly differently by using the sample function. The 2 says create 2 data sets essentially, replacement means we can reset the random sampling across each vector and the probability gives sample the weight of the splits, 2/3 for train, 1/3 for test. 
iris_sample <- sample(2, nrow(scalediris), replace=TRUE, prob=c(0.67, 0.33))
#We then just need to use the new variable to create the test/train outputs, selecting the first four rows as they are the numeric data in the iris data set and we want to predict Species (https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/sample)

View(iris)

View(iris_training)

View(iris_sample)

iris_training <- scalediris[iris_sample==1, 1:4]
iris_test <- scalediris[iris_sample==2, 1:4]
#Now we need to create our 'Y' variables or labels need to input into the KNN function
iris.trainLabels <- iris[iris_sample==1, 5]
iris.testLabels <- iris[iris_sample==2, 5]
#So now we will deploy our model 

iris_pred <- knn(train = iris_training, test = iris_test, cl=iris.trainLabels, k=3, prob = TRUE)#probabilities are a percentage of points per class for each point, (kNN equals 4 for example and 3 of 4 are blue then 75% chance of being blue)

View(iris_pred)

View(attr(iris_pred, "prob"))

library(gmodels)
IRISPREDCross <- CrossTable(iris.testLabels, iris_pred, prop.chisq = FALSE)
#Looks like we got all but three correct, not bad


#You can also use caret for KNN, but it's not as specialized as the above, but but does have some additional capabilities for evaluation. 

```

## Example with Caret using 10-k cross-validation 

```{r}

set.seed(1981)
scalediris$Species <- iris$Species #adding back in the label for caret

iris_training_car <- scalediris[iris_sample==1, 1:5]  
iris_test_car <- scalediris[iris_sample==2, 1:5]

trctrl <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 3) # generic control to pass back into the knn mode using the cross validation method. 

iris_knn <- train(Species~.,
                  data = iris_training_car,
                  method="knn",
                  tuneLength=10,
                  trControl= trctrl,#cv method above, will select the optimal K
                  preProcess="scale") #already did this but helpful reference

iris_knn#take a look

plot(iris_knn)#can also plot

varImp(iris_knn)#gives us variable importance on a range of 0 to 100

iris_pred <- predict(iris_knn, iris_test_car)

iris_pred #gives a character predicted value for each row.

confusionMatrix(iris_pred, iris_test_car$Species)

table(iris_test_car$Species)#looks like we mis-classified 3 virginica as versicolor



```


# In Class Exercise 
```{r}

# Using the bank dataset build a kNN model with your groups recommended k value and compare to the example from above, use the tuning data. Then use the test data and see how the model preforms. 

# Also think about this questions, given what you know about kNN do you really need to do a robust training effort? 


```



