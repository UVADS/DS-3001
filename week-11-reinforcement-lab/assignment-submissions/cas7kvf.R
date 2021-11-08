library(e1071)
library(tidyverse)
library(plotly)
library(htmltools)
library(devtools)
library(caret)
library(NbClust)

#Something I haven't done before: deciding, building, and evaluating a model 
#on data that I did not know the context of.

df <- read_csv('/Users/catherineschuster/Desktop/Fall 2021/DS 3001/DS-3001-Main/week-11-reinforcement-lab/data-frame.csv')
str(df)

ggplot(df, aes(x=cora, y=corc))+geom_point()


str(df)

clust_data = df[, c('cora', 'corc')]

#Normalizing numerical values in the clustering data.

normalize <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

num_cols <- names(select_if(clust_data, is.numeric))

clust_data[num_cols] <- as_tibble(lapply(clust_data[num_cols], normalize))


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
elbow_data = data.frame(k = 1:10, explained_var)

ggplot(elbow_data, 
       aes(x = k,  
           y = explained_var)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1) +            #<- sets the thickness of the line
  xlab('k') + 
  ylab('Inter-cluster Variance / Total Variance') + 
  theme_light() 




set.seed(17)
kmeans_5 = kmeans(clust_data, centers = 5, algorithm = "Lloyd")

#Evaluate the quality of the clustering 
betweenss_5 = kmeans_5$betweenss

# Total variance, "totss" is the sum of the distances between all the points in the data set.
totss_5 = kmeans_5$totss

# Variance accounted for by clusters.
(var_exp_5 = betweenss_5 / totss_5)


df$cluster <- kmeans_5$cluster

df


fig <- plot_ly(df, 
               type = "scatter",
               mode="markers",
               symbol = ~cluster,
               color = ~cluster,
               colors = c('red', 'blue', 'green', 'pink', 'yellow'),
               x = ~cora, 
               y = ~corc)

fig
