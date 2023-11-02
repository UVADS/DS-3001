

#%%
# Load libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
#%%
import os
os.listdir()
#print(os.getcwd())
os.chdir('c:\\Users\\Brian Wright\\Documents\\3001Python\\DS-3001')
#%%
# Load Data
house_votes_Dem = pd.read_csv("data/house_votes_Dem.csv", encoding='latin')
house_votes_Rep = pd.read_csv("data/house_votes_Rep.csv")
#%%
#Let's take a look at the data
print(house_votes_Dem.head())
house_votes_Dem.info()
#%%
#Create summary of aye and nay votes by party label
house_votes_Rep.groupby("party.labels").agg({"aye": "sum", "nay": "sum", "other": "sum"})
#house_votes_Dem.groupby("party.labels").agg({"aye": "sum", "nay": "sum", "other": "sum"})

#%%
# Step 2: run k-means
clust_data_Dem = house_votes_Dem[["aye", "nay", "other"]]
kmeans_obj_Dem = KMeans(n_clusters=2, random_state=1).fit(clust_data_Dem)

#%%
#Take a look at the clustering results
print(kmeans_obj_Dem.cluster_centers_)
print(kmeans_obj_Dem.labels_)
print(kmeans_obj_Dem.inertia_)


#%%
#create a 3d plot of the data of the clusters
fig = px.scatter_3d(house_votes_Dem, x="aye", y="nay", z="other", color=kmeans_obj_Dem.labels_,
                    title="Aye vs. Nay vs. Other votes for Democrat-introduced bills")
fig.show(renderer="browser")

#%%
#calculate the within cluster sum of squares
wcss = []
for i in range(1, 11):
    kmeans_obj_Dem = KMeans(n_clusters=i, random_state=1).fit(clust_data_Dem)
    wcss.append(kmeans_obj_Dem.inertia_)


#%%    
# Plotting the graph
elbow_data_Dem = pd.DataFrame({"k": range(1, 11), "wcss": wcss})
fig = px.line(elbow_data_Dem, x="k", y="wcss", title="Elbow Method")
fig.show()
#%%
#Retrain the model with 3 clusters
kmeans_obj_Dem = KMeans(n_clusters=3, random_state=1).fit(clust_data_Dem)

#%%
#Further assess the model using total variance explained
#calculate total variance explained
total_sum_squares = np.sum((clust_data_Dem - np.mean(clust_data_Dem))**2)
total = np.sum(total_sum_squares)
print(total)

#%%
between_SSE = (total-kmeans_obj_Dem.inertia_)
print(between_SSE)
Var_explained = between_SSE/total
print(Var_explained)

#%%
# Step 3: visualize plot
fig = px.scatter_3d(house_votes_Dem, x="aye", y="nay", z="other",color="party.labels", symbol=kmeans_obj_Dem.labels_,
                    title="Aye vs. Nay vs. Other votes for Democrat-introduced bills")
# add x for center of the cluster
fig.add_trace(go.Scatter3d(x=kmeans_obj_Dem.cluster_centers_[:, 0], y=kmeans_obj_Dem.cluster_centers_[:, 1],
                            z=kmeans_obj_Dem.cluster_centers_[:, 2], mode="markers", marker=dict(size=20, color="black"),
                            name="Cluster Centers"))

fig.show(renderer="browser")

#%%




#%%
#Another method using the silhouette score to determine the number of clusters
#What is a silhouette score? https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
# Silhouette measures BOTH the separation between clusters AND cohesion in respective clusters. 
# Intuitively speaking, it is the difference between separation B (average distance between each point and all points of its nearest cluster) 
# and cohesion A (average distance between each point and all other points in its cluster) divided by max(A,B).
# It is a value between -1 and 1, the higher the better (negative value means that the point is more closer to 
# the nearest cluster than to its own, which is quite a problem).



#%%
from sklearn.metrics import silhouette_score

# Run NbClust
silhouette_scores = []
for k in range(2, 11):
    kmeans_obj = KMeans(n_clusters=k, algorithm="auto", random_state=1).fit(clust_data_Dem)
    silhouette_scores.append(silhouette_score(clust_data_Dem, kmeans_obj.labels_))

best_nc = silhouette_scores.index(max(silhouette_scores))+2

#%%
#plot the silhouette scores
fig = go.Figure(data=go.Scatter(x=list(range(2, 11)), y=silhouette_scores))
fig

#%%
# evaluate the kmeans model using gap statistic
#what is gap statistic? https://web.stanford.edu/~hastie/Papers/gap.pdf


#%%
# Decision Tree model using clusters
kmeans_obj_Dem = KMeans(n_clusters=3, algorithm="auto", random_state=1).fit(clust_data_Dem)
house_votes_Dem['clusters'] = kmeans_obj_Dem.labels_

tree_data = house_votes_Dem.drop(columns=["Last.Name"])
tree_data[['party.labels', 'clusters']] = tree_data[['party.labels', 'clusters']].astype('category')

train, tune_and_test = train_test_split(tree_data, test_size=0.3, random_state=1)
tune, test = train_test_split(tune_and_test, test_size=0.5, random_state=1)

features = train.drop(columns=["party.labels"])
target = train["party.labels"]

party_dt = DecisionTreeClassifier(random_state=1)
party_dt.fit(features, target)

dt_predict_1 = party_dt.predict(tune.drop(columns=["party.labels"]))
print(confusion_matrix(dt_predict_1, tune["party.labels"]))

# Without clusters
tree_data_nc = tree_data.drop(columns=["clusters"])
train, tune_and_test = train_test_split(tree_data_nc, test_size=0.3, random_state=1)
tune, test = train_test_split(tune_and_test, test_size=0.5, random_state=1)

features = train.drop(columns=["party.labels"])
target = train["party.labels"]

party_dt = DecisionTreeClassifier(random_state=1)
party_dt.fit(features, target)

dt_predict_t = party_dt.predict(tune.drop(columns=["party.labels"]))
print(confusion_matrix(dt_predict_t, tune["party.labels"]))
# %%
