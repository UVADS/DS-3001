---
title: "Reinforcement Lab"
author: "Nathan Patton"
date: "11/3/2021"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r}
library(tidyverse)
library(plotly)
library(htmltools)
library(devtools)
library(caret)
library(NbClust)
```

# read in data and create dataframe (df1)
```{r}
df <- read_csv("C:/Users/natha/OneDrive/Documents/DS-3001-main/week-11-reinforcement-lab/data-summary.csv")
view(df)
df1 <- select(df,main_colors,opp_colors,on_play,num_turns,won)
```

# feature engineering (cora,corc)
```{r}
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
```

# add new features to dataframe
```{r}
df1 <- df1 %>% mutate(cora = vec1)
df1 <- df1 %>% mutate(corc = vec3)
```

# make scatter plot comparing new features
```{r}
clust_data = df[, c("num_turns")]
View(clust_data)
clust_data <- na.omit(clust_data)

set.seed(1)
kmeans_obj= kmeans(clust_data, centers = 4, 
                        algorithm = "Lloyd")
kmeans_obj

head(kmeans_obj)

kmeans_obj
clusters = as.factor(kmeans_obj$cluster)

View(clusters)

ggplot(df, aes(x = vec1, 
                            y = vec3,
                            color = `num_turns`,
                            shape = clusters)) + 
  geom_point(size = 6) +
  ggtitle("Corc vs Cora") +
  xlab("Cora") +
  ylab("Corc") +
  scale_shape_manual(name = "Cluster", labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"), values = c("1", "2", "3", "4")) + theme_light()

```

## Determing the Variance of Different Amounts of Clusters Used 
```{r,echo=TRUE}
#Use the function we created to evaluate several different number of clusters

# Run an algorithm with 3 centers.
set.seed(1)
kmeans_obj = kmeans(clust_data, centers = 3, algorithm = "Lloyd")

# Inter-cluster variance.
num3 = kmeans_obj$betweenss

# Total variance.
denom3 = kmeans_obj$totss

# Variance accounted for by clusters.
(var_exp3 = num3 / denom3)

# variance goes up when using 3 clusters 

# The function explained_variance wraps our code for calculating 
# the variance explained by clustering.
explained_variance = function(data_in, k){
  
  # Running the kmeans algorithm.
  set.seed(1)
  kmeans_obj = kmeans(data_in, centers = k, algorithm = "Lloyd", iter.max = 30)
  
  # Variance accounted for by clusters:
  # var_exp = intercluster variance / total variance
  var_exp = kmeans_obj$betweenss / kmeans_obj$totss
  var_exp 
}
  
# Recall the variable we are using for the data that we're clustering.
View(clust_data)

# The sapply() function plugs in several values into our explained_variance function.
#sapply() takes a vector, lapply() takes a dataframe
explained_var = sapply(1:10, explained_variance, data_in = clust_data)

View(explained_var)


# Data for ggplot2.
elbow_data = data.frame(k = 1:10, explained_var)
View(elbow_data)
```

## Elbow Chart for Varying Amounts of Clusters
```{r,echo=TRUE}
#Create a elbow chart of the output '

# Plotting data.
ggplot(elbow_data, 
       aes(x = k,  
           y = explained_var)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1) +            #<- sets the thickness of the line
  xlab('k') + 
  ylab('Inter-cluster Variance / Total Variance') + 
  theme_light()
```

## Using NBClust to Determine the Number of Clusters that Should be Used

```{r}
#Use NbClust to select a number of clusters

# Run NbClust
(nbclust_obj = NbClust(data = clust_data, method = "kmeans"))

# View the output that shows the number of clusters each method recommends.
View(nbclust_obj$Best.nc)

# Subset the 1st row from Best.nc and convert it 
# to a data frame so ggplot2 can plot it.
freq_k = nbclust_obj$Best.nc[1,]
freq_k = data.frame(freq_k)
View(freq_k)

# Check the maximum number of clusters suggested.
max(freq_k)

```

## Histogram Cluster Analysis 
```{r,echo=TRUE}
#Display the results visually 

# Plot as a histogram.
ggplot(freq_k,
       aes(x = freq_k)) +
  geom_bar() +
  scale_x_continuous(breaks = seq(0, 15, by = 1)) +
  scale_y_continuous(breaks = seq(0, 8, by = 1)) +
  labs(x = "Number of Clusters",
       y = "Number of Votes",
       title = "Cluster Analysis")

# Cluster Analysis plot shows the number of clusters that should be used 

```
