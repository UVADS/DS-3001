library(tidyverse)

# read in data and create dataframe (df1)
df <- read_csv("/Users/audreyhimes/Documents/UVA F21/DS 3001/DS-3001-new/week-11-reinforcement-lab/data-summary.csv")

# only selecting a few of the variables (because df is so big)
df1 <- select(df,main_colors,opp_colors,on_play,num_turns,won)


# feature engineering (cora,corc)
df2 <- select(df,"deck_Adeline, Resplendent Cathar":"deck_Wrenn and Seven")
  # from df, select a few columns that we want to see, make them into a matrix
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
  # cora and corc are a few correlations

# make scatter plot comparing new features
ggplot(df1,aes(x=cora,y=corc))+geom_point()
  # plotting df1, the two correlation variables, scatterplot
  # do we see clusters?

# Exercise 4
clust_data = df1[, c("cora", "corc")]
View(clust_data)



### creating an elbow plot to find optimal k

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
# sapply works like a nested for loop
# will pass in k to our function, will use the given data every single time
# iterates between 1 through 10 clusters

View(explained_var)

# Data for ggplot2.
elbow_data = data.frame(k = 1:10, explained_var)
# creating the dataframe that has number of ks and the variance explained
View(elbow_data)

# Plotting data.
ggplot(elbow_data, 
       aes(x = k,  
           y = explained_var)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1) +            #<- sets the thickness of the line
  xlab('k') + 
  ylab('Inter-cluster Variance / Total Variance') + 
  theme_light()
###

set.seed(1984)

kmeans_obj = kmeans(clust_data, centers = 5, 
                        algorithm = "Lloyd")  


# visualizing the clusters
clusters = as.factor(kmeans_obj$cluster)

# What does the kmeans_obj look like?
View(clusters)


ggplot(df1, aes(x = cora, 
                            y = corc,
                            color = won,  #<- tell R how to color the data points
                            shape = clusters)) + 
  geom_point(size = 6) +
  ggtitle("clustering won") +
  xlab("cora") +
  ylab("corc") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"),
                     values = c("1", "2", "3", "4", "5")) +
  scale_color_manual(name = "Won",         #<- tell R which colors to use and which labels to include in the legend
                     labels = c("TRUE", "FALSE"),
                     values = c("dark green", "red")) +
  theme_light()

ggsave("corr_clustering.png", 
       width = 10, 
       height = 5.62, 
       units = "in")
