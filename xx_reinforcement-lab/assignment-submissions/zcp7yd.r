# Zoe Pham (zcp7yd)

# Reinforcement Lab

# Purpose: I collaborated on this code with my group to learn about other
# ways to apply Clustering or K-Means and understand what it tells us
# about this dataset. More specifically, I used clustering based on number
# of turns, on play, and if the player won and colored the data based on
# main colors to learn about how these factors affect Magic the Gathering
# player wins.

# After class, I looked up more about Magic the Gathering to better 
# understand how these factors affect gameplay and success, as well as 
# real life case studies of clustering like classifying pokemon as
# worthy of catching based on all their stats:

# https://towardsdatascience.com/unsupervised-learning-clustering-60f13b4c27f1library(tidyverse)
library(plotly)
library(htmltools)
library(devtools)
library(caret)
library(NbClust)


# Create dataframe with selected columns
df <- read_csv("C:/Users/zoe0p/OneDrive/Desktop/DS-3001-alonzi/week-11-reinforcement-lab/data-summary.csv")
df1 <- select(df, main_colors, opp_colors, on_play, num_turns, won)

# Sparse matrix: matrix filled with majority 0s

# Feature engineering, mutate (cora, corc)

# select columns we care about
df2 <- select(df,"deck_Adeline, Resplendent Cathar":"deck_Wrenn and Seven")

# create a matrix and vectors
mat = data.matrix(df2) 
vec1 <- vector()
vec3 <- vector()
for(i in 1:nrow(mat) ){
  x<-cor( mat[1,] , mat[i,])
  vec1 <- c(vec1,x)
  z<-cor( mat[47,] , mat[i,])
  vec3 <- c(vec3,z)
}

# add new features to dataframe; correlation a and correlation c
df1 <- df1 %>% mutate(cora = vec1)
df1 <- df1 %>% mutate(corc = vec3)
# data-frame is finished version of this processing

# make scatter plot comparing new features
ggplot(df1,aes(x=cora,y=corc))+geom_point()

# Define columns to cluster
df1 <- na.omit(df1)
clust_data <- df1[, c("on_play", "num_turns", "won")]
View(clust_data)
set.seed(1)
kmeans_obj_data = kmeans(clust_data, centers = 2, algorithm = "Lloyd") 
kmeans_obj_data
clusters = as.factor(kmeans_obj_data$cluster)


# What does the kmeans_obj look like?
View(clusters)


(nbclust_obj = NbClust(data = clust_data, method = "kmeans"))

# View the output of NbClust.
nbclust_obj

# View the output that shows the number of clusters each method recommends.
View(nbclust_obj$Best.nc)

ggplot(df1, aes(x = num_turns, 
                            y = won,
                            shape = clusters)) + 
  geom_point(size = 6) +
  ggtitle("Number of Turns vs. Won") +
  xlab("Number of Turns") +
  ylab("Won") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2"),
                     values = c("1", "2")) +
  theme_light()
