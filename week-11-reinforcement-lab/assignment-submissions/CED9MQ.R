## Week 11 Reinforcement Lab
## Claire Dozier (ced9mq) 
## November 3, 2021 

# I took a clustering approach to begin to understand the data. 

library(tidyr)
library(dplyr)
library(ggplot2)
library(tidyverse)

data <- read.csv("data-frame.csv")

table(data$won)

ggplot(data, aes(x = cora, y = corc, color = won, alpha = 0.2)) + geom_point()

ggplot(data, aes(x = cora, y = corc, color = main_colors, alpha = 0.2)) + geom_point()

### Clustering 
clust_data_num <- data[, c("num_turns", "cora", "corc")]

# Normalize the data
normalize <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

# Normalize the numeric features 
norm_clust_features <- as_tibble(lapply(clust_data_num, normalize))

set.seed(1)
kmeans_obj_mystery = kmeans(norm_clust_features, centers = 5, 
                        algorithm = "Lloyd", iter.max = 50) 


mystery_clusters <- as.factor(kmeans_obj_mystery$cluster) 

ggplot(data, aes(x = cora, 
                 y = corc,
                 shape = mystery_clusters,
                 color = won)) + 
  geom_point(size = 6) +
  ggtitle("Mystery Data Clusters") +
  xlab("cora") +
  ylab("corc") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"),
                     values = c("1", "2", "3", "4", "5")) + # for the legend 
  theme_light()

ggplot(data, aes(x = cora, 
                 y = corc,
                 shape = mystery_clusters,
                 color = num_turns)) + 
  geom_point(size = 6) +
  ggtitle("Mystery Data Clusters") +
  xlab("cora") +
  ylab("corc") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"),
                     values = c("1", "2", "3", "4", "5")) + # for the legend 
  theme_light()

comb_cluster <- cbind(data, mystery_clusters)

summary <- comb_cluster %>% group_by(mystery_clusters) %>% count(won)
ggplot(summary, aes(x = mystery_clusters, y = n, color = won, fill = won)) + geom_bar(stat = "identity", position = "dodge")


# How good is the clustering? 

# Inter-cluster variance,
# "betweenss" is the sum of the distances between points 
# from different clusters.
num_mys = kmeans_obj_mystery$betweenss

# Total variance, "totss" is the sum of the distances
# between all the points in the data set.
denom_mys = kmeans_obj_mystery$totss

# Variance accounted for by clusters.
(var_exp_mys = num_mys / denom_mys)

## Build an elbow plot to examine the variance
explained_variance = function(data_in, k){
  
  # Running the kmeans algorithm.
  set.seed(1)
  kmeans_obj = kmeans(data_in, centers = k, algorithm = "Lloyd", iter.max = 50)
  
  # Variance accounted for by clusters:
  # var_exp = intercluster variance / total variance
  var_exp = kmeans_obj$betweenss / kmeans_obj$totss
  var_exp  
}

explained_var_mys = sapply(1:10, explained_variance, data_in = norm_clust_features)

# Data for ggplot2.
elbow_data_mys = data.frame(k = 1:10, explained_var_mys)
View(elbow_data_mys)

# Plotting data.
ggplot(elbow_data_mys, 
       aes(x = k,  
           y = explained_var_mys)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1) +            #<- sets the thickness of the line
  xlab('k') + 
  ylab('Inter-cluster Variance / Total Variance') + 
  theme_light() 

# It looks like 5 clusters is a good point to stop, because there isn't much explained
# variance increase beyond k = 5. 


