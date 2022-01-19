library(tidyverse)

# read in data and create dataframe (df1)
df <- read_csv("week-11-reinforcement-lab/data-summary.csv")
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
ggplot(df1,aes(x=cora,y=corc))+geom_point(color= 'blue')

### Clustering

clust_data = df1[, c("cora", "corc")]

set.seed(1)
kmeans_obj = kmeans(clust_data, centers = 7, 
                        algorithm = "Lloyd")  

head(kmeans_obj)

### Running NBClust says to use 7 clusters

library(NbClust)

(nbclust_obj = NbClust(data = clust_data, method = "kmeans"))

nbclust_obj

View(nbclust_obj$Best.nc)


### Plot

cor_clusters = as.factor(kmeans_obj$cluster)

ggplot(clust_data, aes(x = corc, 
                            y = cora,
                            shape = cor_clusters, color="values")) + 
  geom_point(size = 6) +
  ggtitle("Cluster Graph") +
  xlab("corc") +
  ylab("cora") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7"),
                     values = c("1", "2", "3", "4", "5", "6", "7")) +
  scale_color_manual(name = "Clusters",         
                     labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7"),
                     values = c("blue", "red", "green", "orange", "black", "purple", "yellow"))+
  theme_light()
