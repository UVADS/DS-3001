# %% [markdown]
# # Decision Trees

# %%
#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz 

# %% [markdown]
# ### CART Example using Sklearn: Use a new Dataset, complete preprocessing, use three data
# ### partitions: Training, Tuning and Testing, and build a model!

# %%
#Read in the data from the github repo, you should also have this saved locally...
winequality = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3001/main/data/winequality-red-ddl.csv")

# %%
#Let's take a look...
print(winequality.info())
print(winequality.head())


# %% [markdown]
# ## Preprocessing

# %%
#drop qaulity column since it predicts text_rank perfectly
winequality= winequality.drop(columns='quality')

# %% [markdown]
# ## Missing Data 

# %%
#Let's see if we have any NA's
print(winequality.isna().sum()) #show location of NA's by variable

# %%
#Let's just drop them
winequality= winequality.dropna()

# %%
print(winequality.info()) #Lost some rows, but should be fine with this size dataset

# %% [markdown]
# ## Collapsing the target

# %%
#Let's collapse text_rank now into only two classes
print(winequality.text_rank.value_counts()) #What should we combine?

# %%
#Condense everything into either average or excellent... good goes to excellent, rest is average
winequality["text_rank"]= winequality["text_rank"].replace(['good','average-ish','poor-ish','poor'], ['excellent','ave','ave','ave'])
print(winequality["text_rank"].value_counts()) #Great!

# %%
#check the prevalence
print(203/(1279+203)) #of excellent

# %%
#Before we start move forward, we have one more preprocessing step
#We must encode text_rank to become a continuous variable as that is the only type sklearn decision trees can currently take
#winequality[["text_rank"]] = OrdinalEncoder().fit_transform(winequality[["text_rank"]])
#print(winequality["text_rank"].value_counts()) #nice

# %% [markdown]
# ## Splitting the Data

# %%
#split independent and dependent variables 
X= winequality.drop(columns='text_rank')
y= winequality.text_rank

# %%
#There is not a easy way to create 3 partitions using the train_test_split
#so we are going to use it twice. Mostly because we want to stratify on the variable we are working to predict. What does that mean?  

#what should I stratify by??? Our Target!
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, stratify= y, random_state=21)
X_tune, X_test, y_tune, y_test = train_test_split(X_test,y_test,  train_size = 0.50,stratify= y_test, random_state=49)


# %% [markdown]
# ## Let's Build the Model 

# %%
#Three steps in building a ML model
#Step 1: Cross validation process- the process by which the training data will be used to build the initial model must be set. As seen below:

kf = RepeatedStratifiedKFold(n_splits=10,n_repeats =5, random_state=42)
# number - number of folds
# repeats - number of times the CV is repeated, takes the average of these repeat rounds

# This essentially will split our training data into k groups. For each unique group it will hold out one as a test set
# and take the remaining groups as a training set. Then, it fits a model on the training set and evaluates it on the test set.
# Retains the evaluation score and discards the model, then summarizes the skill of the model using the sample of model evaluation scores we choose

# %%
#What score do we want our model to be built on? Let's use:
#AUC for the ROC curve - remember this is measures how well our model distinguishes between classes
#Recall - this is sensitivity of our model, also known as the true positive rate (predicted pos when actually pos)
#Balanced accuracy - this is the (sensitivity + specificity)/2, or we can just say it is the number of correctly predicted data points
print(metrics.SCORERS.keys()) #find them

# %%
#Define score, these are the keys we located above. This is what the models will be scored by
scoring = ['roc_auc','recall','balanced_accuracy']

# %%
#Step 2: Usually involves setting a hyper-parameter search. This is optional and the hyper-parameters vary by model. 
#define parameters, we can use any number of the ones below but let's start with only max depth
#this will help us find a good balance between under fitting and over fitting in our model

param={"max_depth" : [1,2,3,4,5,6,7,8,9,10,11],
        #"splitter":["best","random"],
        #"min_samples_split":[5,10,15,20,25],
        #"min_samples_leaf":[5,10,15,20,25],
        #"min_weight_fraction_leaf":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        #"max_features":["auto","log2","sqrt",None],
        #"max_leaf_nodes":[10,20,30,40,50],
        #'min_impurity_decrease':[0.00005,0.0001,0.0002,0.0005,0.001,0.0015,0.002,0.005,0.01],
        #'ccp_alpha' :[.001, .01, .1]
           }

# %%
#Step 3: Train the Model

#Classifier model we will use
cl= DecisionTreeClassifier(random_state=1000)

#Set up search for best decisiontreeclassifier estimator across all of our folds based on roc_auc
search = GridSearchCV(cl, param, scoring=scoring, n_jobs=-1, cv=kf,refit='roc_auc')


#%%
#execute search on our training data, this may take a few seconds ...
model = search.fit(X_train, y_train)

# %% [markdown]
# ## Let's see how we did

# %%
#Retrieve the best estimator out of all parameters passed, based on highest roc_auc
best = model.best_estimator_
print(best) #depth of 5, good

# %%
#Creating the decision tree visual for the best estimator 
dot_data = export_graphviz(best, out_file =None,
               feature_names =X.columns, #feature names from dataset
               filled=True, 
                rounded=True, 
                class_names = ['ave','excellent']) #classification labels 
               
graph=graphviz.Source(dot_data)
graph #if chunk doesn't print visual, run this speicfic line alone

# %%
#What about the specific scores (roc_auc, recall, balanced_accuracy)? Let's try and extract them to see what we are working with ...
print(model.cv_results_) #This is a dictionary and in order to extract info we need the keys

# %%
#Which one of these do we need?
print(model.cv_results_.keys()) #get mean_test and std_test for all of our scores, and will need our param_max_depth as well 

# %%
#Let's extract these scores, using our function!

#Scores: 

auc = model.cv_results_['mean_test_roc_auc']
recall= model.cv_results_['mean_test_recall']
bal_acc= model.cv_results_['mean_test_balanced_accuracy']

SDauc = model.cv_results_['std_test_roc_auc']
SDrecall= model.cv_results_['std_test_recall']
SDbal_acc= model.cv_results_['std_test_balanced_accuracy']

#Parameter:
depth= np.unique(model.cv_results_['param_max_depth']).data

#Build DataFrame:
final_model = pd.DataFrame(list(zip(depth, auc, recall, bal_acc,SDauc,SDrecall,SDbal_acc)),
               columns =['depth','auc','recall','bal_acc','aucSD','recallSD','bal_accSD'])

print(final_model.head())
#Let's take a look at the dataframe in our variable explorer as well to get the full dataframe

# %%
#Warning!
#If we used multiple params... you won't be able to get the scores as easily
#Say we wanted to get the scores based on max_depth still, but this time we used the parameter ccp_alpha as well
#Use the np.where function to search for the indices where the other parameter equals their best result, in this say it is .001
#This is an example code to find auc: #model.cv_results_['mean_test_roc_auc'][np.where((model.cv_results_['param_ccp_alpha'] == .001))]
#Essentially takes indeces and resulting scores where the best parameters were used 

# %%
#Check the depth ...
print(plt.plot(final_model.depth,final_model.auc)) #5 does in fact have the highest (best) AUC!

# %% [markdown]
# ## Variable Importance

# %%
#Variable importance for the best estimator, how much weight does each feature have in determining the classification?
varimp=pd.DataFrame(best.feature_importances_,index = X.columns,columns=['importance']).sort_values('importance', ascending=False)
print(varimp)

# %%
#Graph variable importance
plt.figure(figsize=(10,7))
print(varimp.importance.nlargest(7).plot(kind='barh')) #Alcohol has the largest impact by far!

# %% [markdown]
# ## Let's use the model to predict and the evaluate the performance

# %%
#Confusion Matrix time! Passing our tuning data into our best model, let's see how we do ...
print(ConfusionMatrixDisplay.from_estimator(best,X_tune,y_tune, display_labels = ['ave','excellent'], colorbar=False))

# %% [markdown]
# ## Let's play with the threshold, what do we see 

# %%
## Adjust threshold function, we had to make a change here what's different?
def adjust_thres(x, y, z):
    """
    x=pred_probabilities
    y=threshold
    z=tune_outcome
    """
    thres = pd.DataFrame({'new_preds': ['excellent' if i > y else 'ave' for i in x]})
    thres.new_preds = thres.new_preds.astype('category')
    con_mat = metrics.confusion_matrix(z, thres)  
    print(con_mat)



#def adjust_thres(model,X,y_true, thres):
  #model= best estimator, X= feature variables, y_true= target variables, thres = threshold
  #y_pred = (model.predict_proba(X)[:,1] >= thres).astype(np.int32) #essentially changes the prediction cut off to our desired threshold
  #return metrics.ConfusionMatrixDisplay.from_predictions(y_true,y_pred, display_labels = ['ave','excellent'], colorbar=False)

# %%
#Where should we change the threshold to see a significant difference...
#print(pd.DataFrame(model.predict_proba(X_tune)[:,1]).plot.density())

pred_prob = model.predict_proba(X_tune)[:,1]

# %%
#Let's try .1 as our new threshold ...
print(adjust_thres(pred_prob,.4,y_tune)) #Looks like we are getting a lot more false positives, but also a lot more true positives

# %% [markdown]
# ## Accuracy Score

# %%
print(best.score(X_test,y_test)) #Pretty precise, nice!

# %%
print(best.score(X_tune,y_tune)) #same with tuning set

# %% [markdown]
# ## Feature Engineering: 

# %%
#What feature should we look at?
print(winequality.describe())

# %%
# How about total sulfur dioxide ...
print(plt.hist(winequality['total sulfur dioxide'], width=15, bins = 15))

# %%
#Get the five number summary, and some...
print(winequality['total sulfur dioxide'].describe())

# %%
#make a new dataframe so we can preserve our working environment
winequality1 = winequality.copy(deep=True)

# %%
#lump total sulfure dioxide to below and above its median (38)
winequality1['total sulfur dioxide']=pd.cut(winequality1['total sulfur dioxide'],(0,37,300), labels = ("low","high"))

# %%
#transform it to a numeric variable so we can use it in sklearn decision tree
winequality1[["total sulfur dioxide"]] = OrdinalEncoder().fit_transform(winequality1[["total sulfur dioxide"]])

# %%
#Separate features and target 
X1=winequality1.drop(columns='text_rank')
y1=winequality1.text_rank

# %%
#Data splitting
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, train_size=0.70, stratify= y1, random_state=21)
X_tune1, X_test1, y_tune1, y_test1 = train_test_split(X_test1,y_test1,  train_size = 0.50,stratify= y_test1, random_state=49)

# %%
#define search, model, paramameters and scoring will be the same...
search_eng = GridSearchCV(cl, param, scoring=scoring, n_jobs=-1, cv=kf,refit='roc_auc')

#execute search
model_eng = search_eng.fit(X_train1, y_train1)

# %%
#Take a look at our best estimator for depth this time
best_eng = model_eng.best_estimator_ 
print(best_eng) #4, a different depth!

# %%
#Check out the mean and standard deviation test scores again ...
auc_eng = model_eng.cv_results_['mean_test_roc_auc']
recall_eng= model_eng.cv_results_['mean_test_recall']
bal_acc_eng= model_eng.cv_results_['mean_test_balanced_accuracy']

SDauc_eng = model_eng.cv_results_['std_test_roc_auc']
SDrecall_eng= model_eng.cv_results_['std_test_recall']
SDbal_acc_eng= model_eng.cv_results_['std_test_balanced_accuracy']

#Parameter:
depth_eng= np.unique(model_eng.cv_results_['param_max_depth']).data

#Build DataFrame:
final_model_eng = pd.DataFrame(list(zip(depth_eng, auc_eng, recall_eng, bal_acc_eng,SDauc_eng,SDrecall_eng,SDbal_acc_eng)),
               columns =['depth','auc','recall','bal_acc','aucSD','recallSD','bal_accSD'])

print(final_model_eng.head())
#Let's take a look in our variable explorer as well

# %% [markdown]
# # Compare the confusion matrices from the two models

# %%
#First model
print(ConfusionMatrixDisplay.from_estimator(best,X_tune,y_tune,colorbar=False))

# %%
#New engineered model
print(ConfusionMatrixDisplay.from_estimator(best_eng,X_tune1,y_tune1,colorbar=False)) #Better tpr and tnr!

# %% [markdown]
# ## We can also review the variable importance

# %%
#First model
print(pd.DataFrame(best.feature_importances_,index = X.columns,columns=['importance']).sort_values('importance', ascending=False))


# %%
#Engineered model
print(pd.DataFrame(best_eng.feature_importances_,index = X1.columns,columns=['importance']).sort_values('importance', ascending=False))
#Smaller model so less variables are considered in the decision making for better or worse...

# %% [markdown]
# ## Predict with test, how did we do? 

# %%
#Original model
print(ConfusionMatrixDisplay.from_estimator(best,X_test,y_test,colorbar=False))

# %%
#New engineered model
print(ConfusionMatrixDisplay.from_estimator(best_eng,X_test1,y_test1,colorbar=False)) #Ultimately, very similar results

# %% [markdown]
# ## How about accuracy score?

# %%
#Original model
print(best.score(X_test,y_test)) #accuracy 

# %%
#Engineered model
print(best_eng.score(X_test1,y_test1)) #Basically the same

# %% [markdown]
# ## Another example with a binary dataset this time!

# %%
#Load in new dataset
pregnancy = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3001/main/data/pregnancy.csv")

# %%
#Let's get familiar with the data
print(pregnancy.info())
print(pregnancy.head())

# %%
#We want to build a classifier that can predict whether a shopper is pregnant
#based on the items they buy so we can direct-market to that customer if possible. 

print(pregnancy.PREGNANT.sum())
print(len(pregnancy.PREGNANT))

print((1- pregnancy.PREGNANT.sum()/len(pregnancy.PREGNANT)))
#What does .72 represent in this context? Prevalence of not being pregnant

# %% [markdown]
# # reformat for exploration purposes

# %%
#Creating a vertical dataframe for the pregnant variable, just stacking the variables on top of each other. 
#First get the column names of features then use pd.melt based on the Pregnancy variable
feature_cols = pregnancy.drop(columns='PREGNANT').columns
pregnancy_long = df = pd.melt(pregnancy, id_vars='PREGNANT', value_vars=feature_cols,
             var_name='variable', value_name='value')


print(pregnancy_long.head())

# %% [markdown]
# # See what the base rate likihood of pregnancy looks like for each variable

# %%
# Calculate the probability of being pregnant by predictor variable.
# First let's create a new list to store our probability data
data=[]
#loop through features and retrieve probability of pregnancy for whether it is bought or not bought
#Since the data is binary you can take the average to get the probability.
for col in feature_cols:
    x = pregnancy.groupby([col])['PREGNANT'].mean()
    data.extend([[col,0,x[0]],[col,1,x[1]]])
base_rate = pd.DataFrame(data, columns = ['Var', 'Value','prob_pregnant'])
base_rate['prob_not_pregnant']= 1-base_rate.prob_pregnant
print(base_rate)

# %% [markdown]
# # Build  the model 

# %%
#Split between features and target and do a three way split
X_preg=pregnancy.drop(columns='PREGNANT')
y_preg=pregnancy.PREGNANT
X_train_preg, X_test_preg, y_train_preg, y_test_preg = train_test_split(X_preg, y_preg, train_size=0.70, stratify= y_preg, random_state=21)
X_tune_preg, X_test_preg, y_tune_preg, y_test_preg = train_test_split(X_test_preg,y_test_preg,  train_size = 0.50,stratify= y_test_preg, random_state=49)

# %%
#This time we are going to use a different technique to complete our hyper-parameter search

#define our model, still using the same classifier as before
cl2=DecisionTreeClassifier(random_state=1000)

#prune our model using minimal cost-complexity pruning, this is an algorithm used to prune a tree to avoid over-fitting
path = cl2.cost_complexity_pruning_path(X_train_preg, y_train_preg)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# %%
#Let's do some exploration into what cost complexity alpha gives the best results without overfitting or underfitting ...
fig, ax = plt.subplots()
#In the following plot, the maximum effective alpha value is removed, because it is the trivial tree with only one node
fig.set_size_inches(18.5, 10.5, forward=True)
plt.xticks(ticks=np.arange(0.00,0.06,.0025))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
#As alpha increases, more of the tree is pruned, which increases the total impurity of its leaves.
#So we know we want to set a relatively low cp_alpha

# %%
#run through all of our alphas and create fitted decision tree classifiers for them so we can explore even further
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train_preg, y_train_preg)
    clfs.append(clf)


# %%
#we remove the last element in clfs and ccp_alphas, because it is the trivial tree with only one node. 
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# Here we show that the number of nodes and tree depth decreases as alpha increases, makes sense since the tree is getting smaller
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.set_size_inches(18.5, 10.5, forward=True)
plt.xticks(ticks=np.arange(0.00,0.06,.0025))
fig.tight_layout()
#As a result a ccp_alpha of 0 would be huge!

# %%
#When ccp_alpha is set to zero and keeping the other default parameters of DecisionTreeClassifier, 
#the tree overfits, leading to a 100% training accuracy and less testing accuracy. 
#As alpha increases, more of the tree is pruned, thus creating a decision tree that generalizes better. 
#Let's try and find the alpha where we get the highest accuracy for both training and testing data simultaneously 
train_scores = [clf.score(X_train_preg, y_train_preg) for clf in clfs]
test_scores = [clf.score(X_tune_preg, y_tune_preg) for clf in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.xticks(ticks=np.arange(0.00,0.06,.0025))
fig.set_size_inches(18.5, 10.5, forward=True)
plt.show()
#In this example, setting ccp_alpha=0.001 maximizes the testing accuracy and training 

# %%
#Make and train tree with .001 ccp_alp so we can use it!
preg_tree=DecisionTreeClassifier(ccp_alpha= 0.001)
preg_tree.fit(X_train_preg,y_train_preg)
# %%
#let's take a look
dot_data_preg = export_graphviz(preg_tree, out_file =None,
               feature_names =X_preg.columns, #column names of our features
               filled=True, 
                rounded=True,
                class_names=['no','yes']) #classification labels
               
graph_preg=graphviz.Source(dot_data_preg)
graph_preg #run specific line to view
# %% [markdown]
# # Variable Importance

# %%
#Shows the reduction in error provided by including a given variable 
print(pd.DataFrame(preg_tree.feature_importances_,index = X_preg.columns,columns=['importance']).sort_values('importance', ascending=False))

# %% [markdown]
# # Test the accuracy 

# %%
print(preg_tree.score(X_test_preg,y_test_preg)) #84% accurate, nice!

# %%
#Confusion matrix
ConfusionMatrixDisplay.from_estimator(preg_tree,X_test_preg,y_test_preg, colorbar = False)

# %% [markdown]
# ## Hit Rate or True Classification Rate, Detection Rate and ROC

# %%
# The error rate is defined as a classification of "Pregnant" when 
# this is not the case, and vice versa. It's the sum of all the
# values where a column contains the opposite value of the row.
pred_preg = preg_tree.predict(X_test_preg)
tn, fp, fn, tp = metrics.confusion_matrix(y_test_preg,pred_preg).ravel()
# The error rate divides this figure by the total number of data points
# for which the forecast is created.
print("True Error Rate = "+ str((fp+fn)/len(y_test_preg)*100)) #Pretty low, great!

# %%
#Detection Rate is the rate at which the algo detects the positive class in proportion to the entire classification A/(A+B+C+D) where A is poss poss
print("Detection Rate = " +str((tp)/len(y_test_preg)*100)) #want this to be higher but there is only so high it can go...
#In this case the percent not pregnant was 72% so pregnant would be 28%, the closer to that the better

# %%
#Building the evaluation ROC and AUC using the predicted and original target variables 
metrics.RocCurveDisplay.from_predictions(y_test_preg,pred_preg)
#Set labels and midline...
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# %%
#We can adjust using a if else statement and the predicted prob, now we have to be 75% sure to classify as pregnant
pred_adjusted = (preg_tree.predict_proba(X_test_preg)[:,1] >= .75).astype(np.int32)
metrics.RocCurveDisplay.from_predictions(y_test_preg,pred_adjusted)
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# %% [markdown]
# # We can also prune the tree to make it less complex 

# %%
#Set parameters for our model, this time let's use the complexity parameter and deepth 
param_prune={"max_depth" : [1,2,3,4,5,6,7],
        #"splitter":["best","random"],
        #'criterion': ['gini','entropy'],
        #"min_samples_split":[5,10,15,20,25],
        #"min_samples_leaf":[2,4,6,8,10],
        #"min_weight_fraction_leaf":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        #"max_features":["auto","log2","sqrt",None],
        #"max_leaf_nodes":[10,20,30,40,50],
        #'min_impurity_decrease':[0.00005,0.0001,0.0002,0.0005,0.001,0.0015,0.002,0.005,0.01],
        'ccp_alpha' : [.001]
           }
# define search, same model and scoring as before
search_preg = GridSearchCV(cl2, param_prune, scoring=scoring, n_jobs=-1, cv=kf,refit='roc_auc') 
#execute search
model_preg = search_preg.fit(X_train_preg, y_train_preg)

#Let's get our results!
best_preg= model_preg.best_estimator_
print(best_preg) #Now only has a max depth of 7

# %%
print(preg_tree.tree_.max_depth) #A lot better than 14!

# %%
#let's take a look
dot_data_preg1 = export_graphviz(best_preg, out_file =None,
               feature_names =X_preg.columns, #column names of our features
               filled=True, 
                rounded=True,
                class_names=['no','yes']) #classification labels
               
graph_preg1=graphviz.Source(dot_data_preg1)
graph_preg1 #run specific line to view
#A lot more condense!
# %%
print(best_preg.score(X_test_preg,y_test_preg)) #Same accuracy as well, great!


