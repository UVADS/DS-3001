
library(tidyverse)
data <- read_csv('data-frame.csv')

data[,c(1,2,3,5)]<-lapply(data[,c(1,2,3,5)],function(x) as.factor(x))

sp<-ggplot(data, aes(x=cora,y=corc, shape = won, color = won))+geom_point()

sp+scale_color_manual(breaks = c("TRUE","FALSE"),
                      values = c("green","red"))

### Explained Variance: Quality of Clustering

clust_data <-data[,c(4,6,7)]
clust_data[,c(1,2,3)] <-lapply(clust_data[,c(1,2,3)],function(x) scale(x))


explained_variance = function(data_in, k){
  
  # Running the kmeans algorithm.
  set.seed(1)
  kmeans_obj = kmeans(data_in, centers = k, algorithm = "Lloyd", iter.max = 30)
  
  # Variance accounted for by clusters:
  # var_exp = intercluster variance / total variance
  var_exp = kmeans_obj$betweenss / kmeans_obj$totss
  var_exp  
}

#Now feed in a vector of 1 to 10 to test different numbers of clusters
explained_var = sapply(1:10, explained_variance, data_in = clust_data)

#Combine result of function with the k values in a data frame
elbow_data = data.frame(k = 1:10, explained_var)

#Create a elbow chart of the output 
# Plotting data.
ggplot(elbow_data, 
       aes(x = k,  
           y = explained_var)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1) +            #<- sets the thickness of the line
  xlab('k') + 
  ylab('Inter-cluster Variance / Total Variance') + 
  theme_light()

kmeans_obj = kmeans(clust_data, centers = 7, 
                    algorithm = "Lloyd")

clustered = as.factor(kmeans_obj$cluster)

ggplot(data, aes(x = cora, 
                 y = corc,
                 color = won,  #<- tell R how to color 
                 #   the data points
                 shape = clustered)) + 
  geom_point(size = 4) +
  ggtitle("Clustered Data") +
  xlab("Cora") +
  ylab("Corc") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7"),
                     values = c("1", "2","3","4","5","6","7")) +
  scale_color_manual(name = "Winner",         #<- tell R which colors to use and
                     #   which labels to include in the legend
                     labels = c("Win", "Lose"),
                     values = c("green", "red")) +
  theme_light()
