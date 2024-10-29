"""
KNN: Week 6
"""

#%%
# first, import your libraries!
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import metrics
from plot_metric.functions import BinaryClassification #need to pip install plot metric

#%%
# -------- Data prep --------
# Based on the following data summary, what questions and business metric should we use? 

bank_data = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3001/main/data/bank.csv")
print(bank_data.info())

#%%
# now, let's check the data composition
print("marital", bank_data.marital.value_counts())   # 3 levels
print("education", bank_data.education.value_counts())   # 4 levels
print("default", bank_data.default.value_counts())   # 2 levels
print("job", bank_data.job.value_counts())   # 12 levels! What should we do?
print("contact", bank_data.contact.value_counts())   # 3 levels -- difference between cellular and telephone?
print("housing", bank_data.housing.value_counts())   # 2 levels
print("poutcome", bank_data.poutcome.value_counts())   # 4 levels
print("signed up", bank_data['signed up'].value_counts())   # 2 levels

#%%
# Let's collapse `job` which has 12 levels
employed = ['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
           'self-employed', 'services', 'technician']
# unemployed = ['student', 'unemployed', 'unknown']
bank_data.job = bank_data.job.apply(lambda x: "Employed" if x in employed else "Unemployed")
print(bank_data.job.value_counts())

#%%
# now, we convert the appropriate columns into factors
cat = ['job', 'marital', 'education', 'default', 'housing', 'contact',
      'poutcome', 'signed up']   # select the columns to convert
bank_data[cat] = bank_data[cat].astype('category')
bank_data.info()

#%%
# -------- Check for missing data --------

import seaborn as sns
sns.displot(
    data=bank_data.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
# plt.savefig("visualizing_missing_data_with_barplot_Seaborn_distplot.png", dpi=100)
# the above line will same the image to your computer!

# NO MISSING DATA!

#%%
# now, we normalize the numeric variables
numeric_cols = bank_data.select_dtypes(include='int64').columns
print(numeric_cols)


#%%
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
d = scaler.fit_transform(bank_data[numeric_cols])   # conduct data transformation
scaled_df = pd.DataFrame(d, columns=numeric_cols)   # convert back to pd df; transformation converts to array
bank_data[numeric_cols] = scaled_df   # put data back into the main df

#%%
bank_data.describe()   # as we can see, the data is now normalized!

#%%
# Now, we onehot encode the data -- for reference, this is the process of converting categorical variables to a usable form for 
# a machine learning algorithm.

cat_cols = bank_data.select_dtypes(include='category').columns
print(cat_cols)

encoded = pd.get_dummies(bank_data[cat_cols])
encoded.head()   # note the new columns

#What types of variables does the get_dummies function work on and does it have a feature to remove the original column?

#%%
# now we want to drop the old columns we onehot encoded 
bank_data = bank_data.drop(cat_cols, axis=1)

#%%
# and then join them
bank_data = bank_data.join(encoded)
#What is this join function doing? What is it joining on?

#%%
print(bank_data.info())

#%%
# -------- Train model! --------
# check prevalence
print(bank_data['signed up_1'].value_counts()[1] / bank_data['signed up_1'].count())
# This means that at random, we have an 11.6% chance of correctly picking a subscribed individual. Let's see if kNN can do any better.

#%%
"""
X = bank_data.drop(['signed up_1'], axis=1).values   # independent variables
y = bank_data['signed up_1'].values                  # dependent variable
"""

train, test = train_test_split(bank_data,  test_size=0.4, stratify = bank_data['signed up_1']) 
test, val = train_test_split(test, test_size=0.5, stratify=test['signed up_1'])

#%%
# now, let's train the classifier for k=9
import random
random.seed(1984)   # kNN is a random algorithm, so we use `random.seed(x)` to make results repeatable

X_train = train.drop(['signed up_1'], axis=1).values
y_train = train['signed up_1'].values

neigh = KNeighborsClassifier(n_neighbors=9)
neigh.fit(X_train, y_train)

#%%
# now, we check the model's accuracy on the test data:

X_val = val.drop(['signed up_1'], axis=1).values
y_val = val['signed up_1'].values

print(neigh.score(X_val, y_val))

#%%
# now, we test the accuracy on our validation data.

X_test = test.drop(['signed up_1'], axis=1).values
y_test = test['signed up_1'].values

print(neigh.score(X_test, y_test))

#%%
# -------- Evaluate model --------
# A 99.0% accuracy rate is pretty good but keep in mind the baserate is roughly 89/11, so we have more or less a 90% chance of 
# guessing right if we don't know anything about the customer, but the negative outcomes we don't really care about, this model's 
# value is being able to id sign ups when they are actually sign ups. This requires us to know are true positive rate, or 
# Sensitivity or Recall. So let's dig a little deeper.   

# create a confusion matrix
y_val_pred = neigh.predict(X_val)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val,y_val_pred, labels=neigh.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=neigh.classes_)  
disp.plot()
plt.show()


#%%
# create a classification report
# create classification report
from sklearn.metrics import classification_report

print(classification_report(y_val_pred, y_val))

# we didn't get sensitivity and specificity, so we'll calculate that ourselves.
sensitivity = 943/(943+72)   # = TP/(TP+FN)
specificity = 7707/(7707+4)   # = TN/(TN+FP)
print(sensitivity, specificity)

#%%
# ------- Selecting the correct 'k' ---------
def chooseK(k, X_train, y_train, X_test, y_test):
    random.seed(1)
    print("calculating... ", k, "k")    # I'll include this so you can see the progress of the function as it runs
    class_knn = KNeighborsClassifier(n_neighbors=k)
    class_knn.fit(X_train, y_train)
    
    # calculate accuracy
    accu = class_knn.score(X_test, y_test)
    return accu

# We'll test odd k values from 1 to 21. We want to create a table of all the data, so we'll use list comprehension to create 
# the "accuracy" column. 

#%%
# REMEMBER: Python is end-exclusive; we want UP to 21 to we'll have to extend the end bound to include it
test = pd.DataFrame({'k':list(range(1,22,2)), 
                     'accu':[chooseK(i, X_train, y_train, X_test, y_test) for i in list(range(1, 22, 2))]})

#%%
print(test.head())

#%%
test = test.sort_values(by=['accu'], ascending=False)
print(test.head())

#%%
# From here, we see that the best value of k is at the top of the df!

# Let's go through the code we wrote in a bit more detail, specifically regarding the DataFrame construction.

# pandas DataFrames wrap around the Python dictionary data type, which is identifiable by the use of curly brackets ({}) 
# and key-value pairs. The keys correspond to the column names (i.e. 'k' or 'accu') while the values are a list comprised of 
# all the values we want to include. 

# For 'k', we made a list of the range of numbers from 1 to 22 (end exclusive), selecting only every *other* value. This is 
# done using the syntax: `range(first_val, end_val, by=?)`. Having no `by=` value means that we select every value in that range.

# For 'accu', we used list comprehension, which boils down to being loop shorthand with the output being entered into a list.
#  --> for more info on list comprehension, check this link: https://www.w3schools.com/python/python_lists_comprehension.asp

# Now, let's graph our results!
plt.plot(test['k'], test['accu'])
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()

#%%
# -------- Adjusting the threshold --------
# we want to make a table containing: probability, expected, and actual values

test_probs = neigh.predict_proba(X_test)
test_preds = neigh.predict(X_test)

#%%
# convert probabilities to pd df
test_probabilities = pd.DataFrame(test_probs, columns = ['not_signed_up_prob', 'signed_up_prob'])
test_probabilities.head()
#%%
final_model = pd.DataFrame({'actual_class': y_test.tolist(),
                           'pred_class': test_preds.tolist(),
                           'pred_prob': test_probabilities['signed_up_prob']})
final_model.head()

#%%
# convert classes to categories
final_model.actual_class = final_model.actual_class.astype('category')
final_model.pred_class = final_model.pred_class.astype('category')

#%%
# create probability distribution graph
sns.displot(final_model, x="pred_prob", kind="kde")

#%%
# just to see this a little clearer
print(final_model.pred_prob.value_counts())

# In most datasets, the probabilities range between 0 and 1, causing uncertain predictions. A threshold must be set for 
# where you consider the prediction to actually be a part of the positive class. Is a 60% certainty positive? How about 40%? 
# This is where you have more control over your model's classifications. **This is especially useful for reducing incorrect 
# classifications that you may have noticed in your confusion matrix.**

#%%
def adjust_thres(x, y, z):
    """
    x=pred_probabilities
    y=threshold
    z=tune_outcome
    """
    thres = pd.DataFrame({'new_preds': [1 if i > y else 0 for i in x]})
    thres.new_preds = thres.new_preds.astype('category')
    con_mat = confusion_matrix(z, thres)  
    print(con_mat)

#%%
confusion_matrix(final_model.actual_class, final_model.pred_class)   # original model

#%%
adjust_thres(final_model.pred_prob, .90, final_model.actual_class)   # raise threshold 
#%%
adjust_thres(final_model.pred_prob, .3, final_model.actual_class)   # lower threshold

#%%
# -------- More for next week: Model evaluation --------
# ----- ROC/AUC Curve -----
# I'll show you a couple of options! The first is very simple and the other requires another package you can download.

# The first:
# basic graph
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(y_test, final_model.pred_prob)
auc = metrics.roc_auc_score(y_test, final_model.pred_prob)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#%%
# The second:
# run `pip install plot_metric` in your terminal

from plot_metric.functions import BinaryClassification

# Visualisation with plot_metric
bc = BinaryClassification(y_test, final_model.pred_class, labels=["0", "1"])

# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve()
plt.show()

#%%
# ----- F1 Score -----
print(metrics.f1_score(y_test, final_model.pred_class))
#%%
# ----- Log Loss -----
print(metrics.log_loss(y_test, final_model.pred_class))






#%%
# -------- Another quick example! --------
from pydataset import data

iris = data("iris")
print(iris.info())

#%%
from sklearn.preprocessing import scale

cols = list(iris.columns[:4])
scaledIris = pd.DataFrame(scale(iris.iloc[:, :4]), index=iris.index, columns=cols)
scaledIris['Species'] = iris['Species']
print(scaledIris.info())
#%%
# split datasets
irisTrain, irisTest = train_test_split(scaledIris,  test_size=0.4, stratify = scaledIris['Species']) 
irisTest, irisVal = train_test_split(irisTest, test_size=0.5, stratify = irisTest['Species'])

Xi_train = irisTrain.drop(['Species'], axis=1)
yi_train = irisTrain['Species']

Xi_test = irisTest.drop(['Species'], axis=1)
yi_test = irisTest['Species']

Xi_val = irisVal.drop(['Species'], axis=1)
yi_val = irisVal['Species']

#%%
# construct classifier
iris_neigh = KNeighborsClassifier(n_neighbors=3)
iris_neigh.fit(Xi_train, yi_train)

#%%
# look at the scores 
print(iris_neigh.score(Xi_test, yi_test))
print(iris_neigh.score(Xi_val, yi_val))

plot_confusion_matrix(iris_neigh, Xi_val, yi_val, cmap='Blues')  
plt.show()

#%%
# ---------- Example using 10-k cross validation ---------
# construct kfold object -- remember, Python is object-oriented!!
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=12) 

# split data
X_si = scaledIris.drop(['Species'], axis=1)
y_si = scaledIris['Species']

#%%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

cv_neigh = KNeighborsClassifier(n_neighbors=3)   # create classifier
scores = cross_val_score(cv_neigh, X_si, y_si, scoring='accuracy', cv=rkf, n_jobs=-1)   # do repeated cv

print('Accuracy: %.3f (%.3f)' % (scores.mean(), scores.std()))
plt.plot(scores)
#%%
# more complex version so you can create a graph for testing and training accuracy (not built into the previous version)

#Split arrays or matrices into train and test subsets
Xsi_train, Xsi_test, ysi_train, ysi_test = train_test_split(X_si, y_si, test_size=0.20) 
rcv_knn = KNeighborsClassifier(n_neighbors=6)
rcv_knn.fit(Xsi_train, ysi_train)

print("Preliminary model score:")
print(rcv_knn.score(Xsi_test, ysi_test))

no_neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))

for i, k in enumerate(no_neighbors):
    # We instantiate the classifier
    rcv_knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    rcv_knn.fit(Xsi_train, ysi_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = rcv_knn.score(Xsi_train, ysi_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = rcv_knn.score(Xsi_test, ysi_test)

# Visualization of k values vs accuracy
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(no_neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(no_neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#%%
# --------- Variable importance ---------
# There is no easy way in SKLearn to calculate variable importance for a KNN model. So, we'll use a slightly hacked-together 
# solution.

# Variable importance reflects the significance one variable has on the model. If a variable is more important, that variable 
# being removed/permuted has a larger effect on the output of the model. So, if we check the changes such permutations have, we 
# should be able to extract the feature importance.

data = {'sepal_length': [0], 'sepal_width': [0], 'petal_length': [0], 'petal_width': [0]}
feat_imp = pd.DataFrame(data)
feat_imp.head()
#%%
# baseline
fin_knn = KNeighborsClassifier(n_neighbors=7)
fin_knn.fit(Xsi_train, ysi_train)

print(fin_knn.score(Xsi_test, ysi_test))
plot_confusion_matrix(fin_knn, Xsi_test, ysi_test, cmap='Blues')  
#%%
# 1. Change `Sepal.Length`
perm_SL = Xsi_test.copy()   # copy df; we don't want to alter the actual data
perm_SL['Sepal.Length'] = np.random.permutation(perm_SL['Sepal.Length'])   # permute data
perm_SL.head()
#%%
print(fin_knn.score(perm_SL, ysi_test))   # see the new score
#%%
feat_imp['sepal_length'] = fin_knn.score(Xsi_test, ysi_test) - fin_knn.score(perm_SL, ysi_test)   # add to var_imp df
feat_imp.head()
#%%
plot_confusion_matrix(fin_knn, perm_SL, ysi_test, cmap='Blues')    # What got misclassified?
#%%
# Now, doing this over and over is very repetitive (especially if we have a lot of data)
# here's a function!
def featureImportance(X, y, model):
    # create dataframe of variables
    var_imp = pd.DataFrame(columns=list(X.columns))
    var_imp.loc[0] = 0
    base_score = model.score(X, y)
    for col in list(X.columns):
        temp = X.copy()   # # copy df; we don't want to alter the actual data
        temp[col] = np.random.permutation(temp[col])   # permute data
        var_imp[col] = base_score - model.score(temp, y)
        # plot_confusion_matrix(model, temp, y, cmap='Blues')  # what got misclassified?
    print(var_imp)
#%%
featureImportance(Xsi_test, ysi_test, fin_knn)
# from here, we find the important variables!

#%%
# -------- General Eval -------
plot_confusion_matrix(fin_knn, Xsi_test, ysi_test, cmap='Blues')  
#%%
# Looks like we only misclassified one virginica as versicolor. Let's see how certain our predictions were.
iris2_probs = fin_knn.predict_proba(Xsi_test)
print(iris2_probs)



#%%
newlist = [x for x in range(10)] 
# %%
