library(tidyverse)

df <- read_csv("./week-11-reinforcement-lab/data-frame.csv")

ggplot(df,aes(x=cora,y=corc))+geom_point()


set.seed(1984)

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
explained_var <- as.tibble(explained_var)
print(explained_var[5,]*100)

elbow_data = data.frame(k = 1:10, explained_var)
View(elbow_data)
ggplot(elbow_data, 
       aes(x = k,  
           y = explained_var)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1)

clust_data = df[, c("cora", "corc")]
kmeans_obj = kmeans(clust_data, centers = 5, 
                        algorithm = "Lloyd")

cluster <- as.factor(kmeans_obj$cluster)

ggplot(df, aes(x = cora, y = corc, color = cluster))+geom_point()
