---
title: "Reinforcement lab"
author: "Kent Williams"
date: "11/3/2021"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
# Description
This reinforcement lab helped me improve my visualization, normalization, and clustering skills.

# One figure I generated
I generated was a 3d model for clustering

#One thing I looked up after class
One thing that I looked up after class was how clustering is used in finance, since that is something that interests me

```{r}
library(tidyverse)
library(plotly)
library(htmltools)
library(devtools)
library(caret)
library(NbClust)
```


```{r}
# read in data and create dataframe (df1)
df <- read_csv("~/Desktop/DS_3001_Notes/DS-3001 - MAIN REPO/week-11-reinforcement-lab/data-summary.csv")
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
```
```{r}
# Normalize the data:
normalize <- function(x){
 (x - min(x)) / (max(x) - min(x))
}
normalize(df1$corc)
normalize(df1$cora)
num_turns2 = normalize(df1$num_turns)

df1$num_turns2 = normalize(df1$num_turns)


clust_data_Df = df1[, c("corc", "cora", "num_turns2")]

set.seed(1) #tells it where to start 
kmeans_obj_Df = kmeans(clust_data_Df, centers = 2, 
                        algorithm = "Lloyd")

kmeans_obj_Df

```

Visualize the Plots
```{r}
party_clusters_Df <- as.factor(kmeans_obj_Df$cluster) #Changing this to a factor

ggplot(df1, aes(x = corc, 
                            y = cora,
                            shape = party_clusters_Df)) + 
  #all of this is just telling it how we want it to appear
  geom_point(size = 6) +
  ggtitle("corc vs. cora") +
  xlab("corc") +
  ylab("cora") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2"),
                     values = c("1", "2")) +
  theme_light()

# Now a graph with color according to salary
ggplot(df1, aes(x = corc, 
                            y = cora,
                            color = num_turns2,
                            shape = party_clusters_Df))+  #= tell R how to color 
  geom_point(size = 6) +
  ggtitle("corc vs. cora") +
  xlab("corc") +
  ylab("cora") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2"),
                     values = c("1", "2")) +
  theme_light()
```

Step 4: Create A 3D plot so we can actually identify target players
```{r}
#Add the assigned clusters to the main dataset, as a new column
df1$clusters <- (party_clusters_Df)
#Regular expression to remove special characters and only have alphaebtical characters
#df1$Player <- gsub("[^[:alnum:]]", "", NBA_2020_Salaries$Player)

# Now graph the 3D plot
fig = plot_ly(df1, 
               type = "scatter3d",
               mode="markers",
               symbol = ~clusters,
               x = ~corc, 
               y = ~cora, 
               z = ~num_turns2,
               #color = ~num_turns,
               text = ~paste('Won:', won)) #this line creates hovertext,
                                          #so if we hover over it with our mouse it shows information
fig
```
