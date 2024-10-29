library(tidyverse)
library(plotly)
library(htmltools)
library(devtools)
library(caret)
library(NbClust)
library(RColorBrewer)
library(ROCR)
library(MLmetrics)

data_frame$won <- as.numeric(data_frame$won)

#standardize those variables
normalize <- function(x){
  na.omit(x)
  (x - min(x)) / (max(x) - min(x))
}
df1 <- as_tibble(lapply(data_frame[,c(6,7)], normalize))

explained_variance = function(data_in, k){
  set.seed(1)
  kmeans_obj = kmeans(data_in, centers = k, algorithm = "Lloyd", iter.max = 30)
  var_exp = kmeans_obj$betweenss / kmeans_obj$totss
  var_exp  
}

explained_var = sapply(1:10, explained_variance, data_in = df1)

#Create a elbow chart of the output 
elbow_data = data.frame(k = 1:10, explained_var)

ggplot(elbow_data, 
       aes(x = k,  
           y = explained_var)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1) +            #<- sets the thickness of the line
  xlab('k') + 
  ylab('Inter-cluster Variance / Total Variance') + 
  theme_light()

#Run the clustering algo with 5 centers
set.seed(1)
kmeans_obj = kmeans(df1, centers = 5, algorithm = "Lloyd")

num = kmeans_obj$betweenss

denom = kmeans_obj$totss

(var_exp = num / denom)

df1$cluster<-kmeans_obj$cluster

fig <- plot_ly(df1, 
               type = "scatter",
               mode="markers",
               symbol = ~cluster,
               x = df1$cora, 
               y = df1$corc, 
               color = ~cluster,
               colors = c('blue','orange', "green", 'red', 'yellow'))

fig