# (Just created a script for this one so it's not quite as long)

# A single decision tree is very deterministic. There is always an element of 
# random chance in the real world, which you can miss if you're only using 1
# model with a hierarchical order of the rules.

# A way to avoid this potential over-fitting problem is to use many different 
# decision trees that evaluate different subsets of variables on subsets of 
# the data. This approach is called bootstrapping (bagging), and in the case of 
# decision trees, it's implemented with an algorithm called Random Forest.

# Random forests can be used for regression and for assessing proximities of 
# data points in unlabeled data. We'll go over how to do that shortly.

# Install and load the "randomForest" package. Make sure you look at all the 
# functionality the package includes.
#install.packages("randomForest")
library(randomForest)
library(help = randomForest)
library(rio)
library(tidyverse)


# Below is a list of some the settings that you can customize when you run
# random forests. It's important to understand how you can customize the
# algorithm to make sure that your math works well for the data you're working
# with.
# Some additional arguments are listed in the help menu, but are used less frequently.
randomForest(formula,             #<- A formula to solve for when using random forests for 
                                  # classification or regression.  
                                  # If left blank, then a random forest runs as an unsupervised 
                                  # machine learning algorithm.
             x,                   #<- A data frame with variables to be used from the training set.
             y = NULL,            #<- A response vector, used if there's no formula. If a factor, classification is assumed, otherwise regression is assumed. If omitted, randomForest will run in unsupervised mode.
             subset = NULL,       #<- An index vector indicating which rows should be used.
             xtest = NULL,        #<- Data frame containing predictors for the test set.
             ytest = NULL,        #<- Response for the test set.
             ntree = 500,         #<- Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times.
             mtry = 5,            #<- Number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3).
             replace = TRUE,      #<- Should sampled data points be replaced, this should always be TRUE, if possible.
             classwt = NULL,      #<- Priors of the classes. Need not add up to one. Ignored for regression. 
             strata = NULL,       #<- Variable used for stratified sampling.
             sampsize = 100,      #<- Size of sample to draw each time.
             nodesize = 5,        #<- Minimum numbers of data points in terminal nodes.
             maxnodes = NULL,     #<- Maximum number of terminal nodes trees in the forest can have. 
             importance = TRUE,   #<- Should importance of predictors be assessed?, TRUE
             localImp = FALSE,    #<- Should casewise importance measures be computed? (Setting this to TRUE will override importance.)
             proximity = FALSE,    #<- Should a proximity measure between rows be calculated?
             norm.votes = TRUE,   #<- If TRUE (default), the final result of votes are expressed as fractions. If FALSE, raw vote counts are returned (useful for combining results from different runs).
             do.trace = FALSE,    #<- If set to TRUE, give a more verbose output as randomForest is run.
             keep.forest = TRUE,  #<- If set to FALSE, the forest will not be retained in the output object. If xtest is given, defaults to FALSE.
             keep.inbag = TRUE)   #<- Should an n by ntree matrix be returned that keeps track of which samples are in-bag in which trees? 

# Note that proximity between data points is the average number of trees for 
# which the data points occupy the same terminal node.

# localImp or Casewise takes all the trees built not using observation
# i, because it was not selected in the bagging processes, essentially the out
# of bag observation (oob) for i. Then submits (permutes) m (a predictor variable)
# into those oob and calculations the accuracy of those models, does the same
# process without submitting variable m and subtract the differences between the 
# two averages.

#==================================================================================

#### Splitting test and training sets ####

# Let's run the random forest on our pregnancy data.
# Let's subset a sample of 200 data points from our data, which we can 
# use to test out the quality of our model.  
str(pregnancy)
# First create a vector of numbers we'll sample.
pregnancy = import("data/pregnancy.csv", check.names = TRUE, stringsAsFactors = TRUE)

pregnancy_factors = as_tibble(apply(pregnancy,                 #<- the data set to apply the function to
                          2,                         #<- for each column
                          function(x) as.factor(x)))  #<- change each variable to factor
str(pregnancy_factors)

sample_rows = 1:nrow(pregnancy_factors)
sample_rows

View(pregnancy_factors)

# sample() is a randomized function, use set.seed() to make your results reproducible.
set.seed(1984) #sample(x, size, replace = FALSE, prob = NULL)
test_rows = sample(sample_rows,
                   dim(pregnancy)[1]*.10, #start with 10% of our dataset, could do 20%
                   # but random forest does require more training data because of the 
                   # sampling so 90% might be a better approach with this small of a dataset
                   replace = FALSE)# We don't want duplicate samples

str(test_rows)


# Partition the data between training and test sets using the row numbers the
# sample() function selected, using a simplified version for the lab you'll need 
# to create three segments 
pregnancy_train = pregnancy_factors[-test_rows,]
pregnancy_test = pregnancy_factors[test_rows,]

#==================================================================================

#### Splitting test and training sets ####

# Check the output.
str(pregnancy_train)
str(pregnancy_test)


#==================================================================================

####  Build a random forest ####


# The radomForest() function is randomized, so use set.seed() to make 
# your results reproducible. 

#general rule to start with the mytry value is square root of the predictors
dim(pregnancy_train)

mytry_tune <- function(x){
  xx <- dim(x)[2]-1
  sqrt(xx)
}


mytry_tune(pregnancy)

str(pregnancy_train)
       
set.seed(1984)	
pregnancy_RF = randomForest(as.factor(PREGNANT)~.,          #<- Formula: response variable ~ predictors.
                            #   The period means 'use all other variables in the data'.
                            pregnancy_train,     #<- A data frame with the variables to be used.
                            #y = NULL,           #<- A response vector. This is unnecessary because we're specifying a response formula.
                            #subset = NULL,      #<- This is unnecessary because we're using all the rows in the training data set.
                            #xtest = NULL,       #<- This is already defined in the formula by the ".".
                            #ytest = NULL,       #<- This is already defined in the formula by "PREGNANT".
                            ntree = 1000,        #<- Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets classified at least a few times.
                            mtry = 4,            #<- Number of variables randomly sampled as candidates at each split. Default number for classification is sqrt(# of variables). Default number for regression is (# of variables / 3).
                            replace = TRUE,      #<- Should sampled data points be replaced.
                            #classwt = NULL,     #<- Priors of the classes. Use this if you want to specify what proportion of the data SHOULD be in each class. This is relevant if your sample data is not completely representative of the actual population 
                            #strata = NULL,      #<- Not necessary for our purpose here.
                            sampsize = 100,      #<- Size of sample to draw each time.
                            nodesize = 5,        #<- Minimum numbers of data points in terminal nodes.
                            #maxnodes = NULL,    #<- Limits the number of maximum splits. 
                            importance = TRUE,   #<- Should importance of predictors be assessed?
                            #localImp = FALSE,   #<- Should casewise importance measure be computed? (Setting this to TRUE will override importance.)
                            proximity = FALSE,    #<- Should a proximity measure between rows be calculated?
                            norm.votes = TRUE,   #<- If TRUE (default), the final result of votes are expressed as fractions. If FALSE, raw vote counts are returned (useful for combining results from different runs).
                            do.trace = TRUE,     #<- If set to TRUE, give a more verbose output as randomForest is run.
                            keep.forest = TRUE,  #<- If set to FALSE, the forest will not be retained in the output object. If xtest is given, defaults to FALSE.
                            keep.inbag = TRUE)   #<- Should an n by ntree matrix be returned that keeps track of which samples are in-bag in which trees? 

#==================================================================================

####  Random forest output ####

# Look at the output of the random forest.
pregnancy_RF

#==================================================================================

#### Random forest output ####

# This is how you can call up the criteria we set for the random forest:
pregnancy_RF$call

# Call up the confusion matrix and check the accuracy of the model.
pregnancy_RF$confusion
pregnancy_RF_acc = sum(pregnancy_RF$confusion[row(pregnancy_RF$confusion) == 
                                                col(pregnancy_RF$confusion)]) / 
  sum(pregnancy_RF$confusion)

pregnancy_RF_acc
# 0.86ish


#==================================================================================

#### Random forest output ####

# View the percentage of trees that voted for each data point to be in each class.
View(as.data.frame(pregnancy_RF$votes))

# The "predicted" argument contains a vector of predictions for each 
# data point.
View(as.data.frame(pregnancy_RF$predicted))

#==================================================================================

#### Random forest output ####

# The "importance" argument provides a table that includes the importance
# of each variable to the accuracy of the classification.
View(as.data.frame(importance(pregnancy_RF, type = 2, scale = TRUE))) #type 1 is error on oob, 
                                                                      # type 2 is total decrease
# in node impurity as measured by the Gini index, look at the differences, stop by wine for example. 
                                                        # scale divides the measures by 
                                                        # their standard errors

View(as.data.frame(pregnancy_RF$importance)) #all the metrics together,not scaled

# The first 2 columns represent the accuracy decrease for each variable by class.
# The 3rd column is the total mean decrease in accuracy for each variable,
# or in other words, by what percentage will the classification accuracy
# decrease if the variable is not used.
# And the 4th column is the mean decrease in the Gini coefficient for
# each variable.

#==================================================================================

#### Random forest output ####

# The "inbag" argument shows you which data point is included in which trees.
View(as.data.frame(pregnancy_RF$inbag))

bagging <- as.data.frame(pregnancy_RF$inbag)
sum(bagging$V50)

dim(pregnancy_RF$inbag)

#==================================================================================

#### Random forest output ####

# The "err.rate" argument includes a list of the cumulative error rates
# for each tree, by class and in aggregate for data points not 
# included in the tree (OOB).
View(as.data.frame(pregnancy_RF$err.rate))

err.rate <- as.data.frame(pregnancy_RF$err.rate)

View(err.rate)

# The "oob.times" argument includes the number of times that each data point
# is not excluded from trees in the random forest.
View(as.data.frame(pregnancy_RF$oob.times))

rf_density <- density(pregnancy_RF$oob.times)
plot(rf_density)
#==================================================================================

#### Visualize random forest results ####

# Let's visualize the results of the random forest.
# Let's start by looking at how the error rate changes as we add more trees.
pregnancy_RF_error = data.frame(1:nrow(pregnancy_RF$err.rate),
                                pregnancy_RF$err.rate)



colnames(pregnancy_RF_error) = c("Number of Trees", "Out of the Box",
                                 "Not Pregnant", "Pregnant")

# Add another variable that measures the difference between the error rates, in
# some situations we would want to minimize this but need to use caution because
# it could be that the differences are small but that both errors are really high,
# just another point to track. 

pregnancy_RF_error$Diff <- pregnancy_RF_error$Pregnant-pregnancy_RF_error$`Not Pregnant`

View(pregnancy_RF_error)


library(plotly)

rm(fig)
fig <- plot_ly(x=pregnancy_RF_error$`Number of Trees`, y=pregnancy_RF_error$Diff,name="Diff", type = 'scatter', mode = 'lines')
fig <- fig %>% add_trace(y=pregnancy_RF_error$`Out of the Box`, name="OOB_Er")
fig <- fig %>% add_trace(y=pregnancy_RF_error$`Not Pregnant`, name="Not Pregnant")
fig <- fig %>% add_trace(y=pregnancy_RF_error$Pregnant, name="Pregnant")

fig

#==================================================================================

#### Optimize the random forest model ####

# Let's just say we want to do the best we can to label pregnant customers as pregnant
# later we can certainly adjust the threshold but want to tune to optimize the models ability 
# to identify the positive target class. 
View(pregnancy_RF_error)

# When we first sort "pregnancy_RF_error" by the "Pregnant" class then
# by the "min" class we see that a lower number of trees  the number
# that minimizes the error in the positive class, also min for oob error.

#==================================================================================

#### Optimize the random forest model ####

# Let's create a random forest model with 400ish trees.
set.seed(1984)	
pregnancy_RF_2 = randomForest(as.factor(PREGNANT)~.,          #<- formula, response variable ~ predictors.
                              #   the period means 'use all other variables in the data'.
                              pregnancy_train,     #<- A data frame with variables to be used.
                              #y = NULL,           #<- A response vector. This is unnecessary because we're specifying a response formula.
                              #subset = NULL,      #<- This is unneccessary because we're using all the rows in the training data set.
                              #xtest = NULL,       #<- This is already defined in the formula by the ".".
                              #ytest = NULL,       #<- This is already defined in the formula by "PREGNANT".
                              ntree = 1000,          #<- Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets classified at least a few times.
                              mtry = 6,            #<- Number of variables randomly sampled as candidates at each split. Default number for classification is sqrt(# of variables). Default number for regression is (# of variables / 3).
                              replace = TRUE,      #<- Should sampled data points be replaced.
                              #classwt = NULL,     #<- Priors of the classes. We will work through this later. 
                              #strata = NULL,      #<- Not necessary for our purpose here.
                              sampsize = 200,      #<- Size of sample to draw each time.
                              nodesize = 20,        #<- Minimum numbers of data points in terminal nodes.
                              #maxnodes = NULL,    #<- The "nodesize" argument limits the number of maximum splits. 
                              importance = TRUE,   #<- Should importance predictors be assessed?
                              #localImp = FALSE,   #<- Should casewise importance measure be computed? (Setting this to TRUE will override importance.)
                              proximity = FALSE,    #<- Should a proximity measure between rows be calculated?
                              norm.votes = TRUE,   #<- If TRUE (default), the final result of votes are expressed as fractions. If FALSE, raw vote counts are returned (useful for combining results from different runs).
                              do.trace = TRUE,     #<- If set to TRUE, give a more verbose output as randomForest is run.
                              keep.forest = TRUE,  #<- If set to FALSE, the forest will not be retained in the output object. If xtest is given, defaults to FALSE.
                              keep.inbag = TRUE)   #<- Should an n by ntree matrix be returned that keeps track of which samples are in-bag in which trees? 

#==================================================================================

#### Compare random forest models ####

# Let's compare the confusion matrices between our 2 random forest models
# and the associated total error rates.
pregnancy_RF$confusion
pregnancy_RF_2$confusion

# Is the second model better? we are classifying more 1s as 1s. Let's see how well it 
# performs on our test data. To do that we'll use the predict() function
# on the test data set.
View(pregnancy_test)

#==================================================================================

#### Generate predictions with your model ####

pregnancy_predict = predict(pregnancy_RF_2,      #<- a randomForest model
                            pregnancy_test,      #<- the test data set to use
                            type = "response",   #<- what results to produce, see the help menu for the options
                            predict.all = TRUE)
?predict.randomForest

# If predict.all=TRUE, then the returned object is a list of two components: 
# aggregate, which is the vector of predicted values by the forest, 
# and individual, which is a matrix where each column contains 
# prediction by a tree in the forest.
str(pregnancy_predict)

pregnancy_RF_2$confusion #On the oob data from the model
#      0   1 class.error
# 0 1209  86  0.066
# 1  180 325  0.356

pregnancy_RF$confusion
#     0   1 class.error
#0 1234  61  0.04710425
#1  196 309  0.38811881

library(caret)
confusionMatrix(as.factor(pregnancy_predict$aggregate),as.factor(pregnancy_test$PREGNANT),positive = "1", 
                dnn=c("Prediction", "Actual"), mode = "everything")

#==================================================================================

#### Understanding the model ####

# How can we refine the model? First, let's make sure we have a good
# understanding for the structure of the model.
# Let's start by looking at the variables that are most important.
View(as.data.frame(pregnancy_RF_2$importance))

#==================================================================================

#### Visualizing the model ####

# We can visualize the impact of the variables on the accuracy of 
# the model with the varImpPlot() function in the "randomForest" package.
dev.off()
varImpPlot(pregnancy_RF_2,     #<- the randomForest model to use
           sort = TRUE,        #<- whether to sort variables by decreasing order of importance
           n.var = 10,        #<- number of variables to display
           main = "Important Factors for Identifying Pregnant Customers",
           #cex = 2,           #<- size of characters or symbols
           bg = "white",       #<- background color for the plot
           color = "blue",     #<- color to use for the points and labels
           lcolor = "orange")  #<- color to use for the horizontal lines

#==================================================================================

#### Tuning the random forest model ####

# There are a lot of parameters to set for the random forest.
# The package can help you identify the right number of variables to 
# randomly sample as candidates at each split (the mtry parameter)
# with the tuneRF() function.

# The tuneRF() function works like this:
str(pregnancy_train)
pregnancy_train <- as.tibble(lapply(pregnancy_train,as.numeric))
dev.off()

pregnancy_RF_mtry = tuneRF(pregnancy_train[ ,1:15],  #<- data frame of predictor variables
                           pregnancy_train$PREGNANT,   #<- response vector (variables), factors for classification and continuous variable for regression
                           mtryStart = 5,                        #<- starting value of mtry, the default is the same as in the randomForest function
                           ntreeTry = 100,                       #<- number of trees used at the tuning step, let's use the same number as we did for the random forest
                           stepFactor = 2,                       #<- at each iteration, mtry is inflated (or deflated) by this value
                           improve = 0.05,                       #<- the improvement in OOB error must be by this much for the search to continue
                           trace = TRUE,                         #<- whether to print the progress of the search
                           plot = TRUE,                          #<- whether to plot the OOB error as a function of mtry
                           doBest = FALSE)                       #<- whether to create a random forest using the optimal mtry parameter

pregnancy_RF_mtry

# Looking at the OOB table, 5 seems to be the right 
# value to use in our random forest. So no need to adjust.

#==================================================================================


#### Size of the forest ####

# If you want to look at the size of the trees in the random forest, 
# or how many nodes each tree has, you can use the treesize() function.
treesize(pregnancy_RF_2,    #<- the randomForest object to use
         terminal = FALSE)  #<- when TRUE, only the terminal nodes are counted, when FALSE, all nodes are counted

# You can use the treesize() function to create a histogram for a visual presentation.
hist(treesize(pregnancy_RF_2,
              terminal = TRUE), main="Tree Size")
dev.off()

#==================================================================================

#### Tuning the model: the ROC curve ####

# It's OK if we classify as someone who is pregnant as not pregnant, 
# but we can have a big lawsuit on our hands if we falsely predict that 
# someone is pregnant when in reality they are not. 

# Let's see how our relative true positives compare to the false positives.
# To do that we'll need to plot a ROC curve.

# Install pROC package in order to create the ROC curve.
library(pROC)
library(help = "pROC")
preg_roc <- roc(pregnancy_predict$aggregate, as.numeric(pregnancy_test$PREGNANT), plot = TRUE)

# First, create a prediction object for the ROC curve.
# Take a look at the "votes" element from our randomForest function.
View(as.data.frame(pregnancy_RF_2$votes))

# The "1" column tells us what percent of the trees voted for 
# that data point as "pregnant". Let's convert this data set into a
# data frame with numbers so we could work with it.
pregnancy_RF_2_prediction = as.data.frame(as.numeric(as.character(pregnancy_RF_2$votes[,2])))
View(pregnancy_RF_2_prediction)

# Let's also take the actual classification of each data point and convert
# it to a data frame with numbers. R classifies a point in either bucket 
# at a 50% threshold.
pregnancy_train_actual = pregnancy_train$PREGNANT

View(pregnancy_train_actual)

#==================================================================================

####  Tuning the model: the ROC curve ####

# The prediction() function from the ROCR package will transform the data
# into a standardized format for true positives and false positives.
pregnancy_prediction_comparison = prediction(pregnancy_RF_2_prediction,           #<- a list or data frame with model predictions
                                             pregnancy_train_actual)#<- a list or data frame with actual class assignments
View(pregnancy_prediction_comparison)


#==================================================================================

#### Tuning the model: the ROC curve ####

# Create a performance object for ROC curve where:
# tpr = true positive rate.
# fpr = fale positive rate.
pregnancy_pred_performance = performance(pregnancy_prediction_comparison, 
                                         measure = "tpr",    #<- performance measure to use for the evaluation
                                         x.measure = "fpr")  #<- 2nd performance measure to use for the evaluation
View(pregnancy_pred_performance)

#==================================================================================

#### Tuning the model: the ROC curve ####


# Here is what the performance() function does with the outputs of the
# prediction() function.
pregnancy_rates = data.frame(fp = pregnancy_prediction_comparison@fp,  #<- false positive classification.
                             tp = pregnancy_prediction_comparison@tp,  #<- true positive classification.
                             tn = pregnancy_prediction_comparison@tn,  #<- true negative classification.
                             fn = pregnancy_prediction_comparison@fn)  #<- false negative classification.

colnames(pregnancy_rates) = c("fp", "tp", "tn", "fn")

#==================================================================================

#### Tuning the model: the ROC curve ####

View(pregnancy_rates)

# As the rows go down the number of remaining unclassified items in the set decreases.
# The first row is the starting point with the initial counts of the positive and 
# negative value, that's why R adds an extra row to the output .

#==================================================================================

#### Tuning the model: the ROC curve ####

# Now let's calculate the true positive and false positive rates for the classification.
str(pregnancy_rates)
tpr = pregnancy_rates$tp / (pregnancy_rates$tp + pregnancy_rates$fn)
fpr = pregnancy_rates$fp / (pregnancy_rates$fp + pregnancy_rates$tn)

# Compare the values with the output of the performance() function, they are the same.
pregnancy_rates_comparison = data.frame(pregnancy_pred_performance@x.values,
                                        pregnancy_pred_performance@y.values,
                                        fpr,
                                        tpr)
colnames(pregnancy_rates_comparison) = c("x.values","y.values","fpr","tpr") #<- rename columns accordingly.
View(pregnancy_rates_comparison)

#==================================================================================

#### Tuning the model: the ROC curve ####
dev.off() 

# Now plot the results.
plot(fpr,          #<- x-axis value.
     tpr,          #<- y-axis value.
     col = "blue",  #<- color of the line. 
     type = "l")   #<- line type.
grid(col = "black")

#==================================================================================

#### Tuning the model: the ROC curve ####

# The performance() function saves us a lot of time, and can be used directly
# to plot the ROC curve.
plot(pregnancy_pred_performance, 
     col = "red", 
     lwd = 3, 
     main = "ROC curve")
grid(col = "black")


#==================================================================================

#### Tuning the model: the ROC curve ####

# Calculate the area under curve (AUC), which can help you compare the 
# ROC curves of different models for their relative accuracy.
pregnancy_auc_RF = performance(pregnancy_prediction_comparison, 
                               "auc")@y.values[[1]]
pregnancy_auc_RF

# Add the AUC value to the ROC plot.
text(x = 0.5, 
     y = 0.5, 
     labels = paste0("AUC = ", 
                     round(pregnancy_auc_RF,
                           2)))

# You can now tune for forecast for each point, but setting a different threshold
# for how many trees need to vote for a point as "pregnant" in order for you to
# make that class assignment. This is where your domain knowledge becomes critical,
# this is a management decision. How much risk can the business tolerate?

# The closer the ROC curve is to a right angle, the higher the AUC, the more accurate
# the model under different thresholds. The reason data scientists build many 
# different classification models and compare the AUC values, is that they want to 
# make sure that they have built the best possible model using several approaches.


