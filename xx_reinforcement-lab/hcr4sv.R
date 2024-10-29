# Hayden Ratliff, hcr4sv
# Reinforcement lab, 11-3-2021

library(tidyverse)
library(tidyverse)
library(plotly)
library(htmltools)
library(devtools)
library(caret)
library(NbClust)

# read in data and create dataframe (df1)
setwd("C:/School/UVA/! Third Year/Fall Term/DS 3001/DS-3001-work/week11-reinforcement")
df <- read_csv("data-summary.csv")
df1 <- select(df,main_colors,opp_colors,on_play,num_turns,won)

# feature engineering (cora,corc)
df2 <- select(df,"deck_Adeline, Resplendent Cathar":"deck_Wrenn and Seven")
mat = data.matrix(df2)
vec1 <- vector()
vec3 <- vector()
for(i in 1:nrow(mat) ){
  x<-cor( mat[1,] , mat[i,])
  vec1 <- c(vec1,x)
  z<-cor( mat[47,] , mat[i,])
  vec3 <- c(vec3,z)
}

# add new features to dataframe
df1 <- df1 %>% mutate(cora = vec1)
df1 <- df1 %>% mutate(corc = vec3)

# make scatter plot comparing new features
ggplot(df1,aes(x=cora,y=corc))+geom_point()


clust_data = df1[,c("cora", "corc")]
set.seed(1)

#Use the function we created to evaluate several different number of clusters
explained_variance = function(data_in, k){
  
  # Running the kmeans algorithm.
  set.seed(1)
  kmeans_obj = kmeans(data_in, centers = k, algorithm = "Lloyd", iter.max = 30)
  
  # Variance accounted for by clusters:
  # var_exp = intercluster variance / total variance
  var_exp = kmeans_obj$betweenss / kmeans_obj$totss
  var_exp  
}

explained_var = sapply(1:10, explained_variance, data_in = clust_data)

# Data for ggplot2.
elbow_data = data.frame(k = 1:10, explained_var)

ggplot(elbow_data, 
       aes(x = k,  
           y = explained_var)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1) +            #<- sets the thickness of the line
  xlab('k') + 
  ylab('Inter-cluster Variance / Total Variance') + 
  theme_light()

(nbclust_obj = NbClust(data = clust_data, method = "kmeans", max.nc = 10))

freq_k = nbclust_obj$Best.nc[1,]
freq_k = data.frame(freq_k)

# Check the maximum number of clusters suggested.
max(freq_k)

#essentially resets the plot viewer back to default
#dev.off()

# Plot as a histogram.
ggplot(freq_k,
       aes(x = freq_k)) +
  geom_bar() +
  scale_x_continuous(breaks = seq(0, 15, by = 1)) +
  scale_y_continuous(breaks = seq(0, 12, by = 1)) +
  labs(x = "Number of Clusters",
       y = "Number of Votes",
       title = "Cluster Analysis")

# based on nbclust, it seems like the optimal k is 7.

kmeans_optimal = kmeans_obj = kmeans(clust_data, centers = 7, algorithm = "Lloyd", iter.max = 30)
clusters = as.factor(kmeans_optimal$cluster)
clust_data$won <- df1$won
clust_data$cluster <- clusters

ggplot(clust_data, aes(x=cora,
                       y=corc, 
                       color=won, 
                       shape=cluster)) + 
  scale_shape_manual(name = "Cluster",
                     labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4",
                               "Cluster 5", "Cluster 6", "Cluster 7"),
                     values = c("1", "2", "3", "4", "5", "6", "7")) +
  scale_color_manual(name = "Won",
                     labels = c("False", "True"),
                     values = c("red", "dark green")) +
  geom_point(size = 5) +
  ggtitle("Clustering on Correlation with Won")


