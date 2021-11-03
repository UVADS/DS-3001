# Part 1: Review

## Exercise 1: What do you know?
# Recent topics I am comfortable with / feel confident saying "I know" about include:
#   
#   1. Evaluation Metrics
# 
# False Positive Rate / True Positive Rate
# Accuracy
# ROC/AUC
# 
# 2. Cleaning Data

## Exercise 2: What do you want to know?
# My weaknesses in these recent topics include:
#   
#   1. Evaluation Metrics
# 
# LogLoss
# Kappa
# F1
# 
# 2. kNN models, especially when it comes to coding them and distinguishing between the kNN function and the train function with method='kNN'
# 
# 3. Data / Model Correspondence -- I find it difficult to find datasets that fit the type of models we learn about in our in-class code examples 
# 
# 4. Clustering

# Part 2: Explore

## Exercise 3: Let's Get the Ball Rolling

### Demo
#Load in necessary libraries
library(tidyverse)
library(NbClust)

# Load in data
df <- read_csv("C:/Users/Maddie/OneDrive/Desktop/3YEAR/Forked-DS-3001/week-11-reinforcement-lab/data-summary.csv")
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


## Exercise 4: challenge

# Cluster!
normalize <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

str(df1)
df1$num_turns <- normalize(df1$num_turns)
df1$on_play <- as.factor(df1$on_play)
df1$on_play <- recode(df1$on_play,
                      'TRUE' = '1',
                      'FALSE' = '0')

df1$won <- as.numeric(df1$won)
str(df1)

clust_df1 <- df1[, c(3, 4)]

# elbow graph
explained_variance = function(data_in, k){
  set.seed(1)
  kmeans_obj = kmeans(data_in, centers = k, algorithm = "Lloyd", iter.max = 30)
  var_exp = kmeans_obj$betweenss / kmeans_obj$totss
  var_exp  
}

explained_var = sapply(1:10, explained_variance, data_in = clust_df1)
explained_var

elbow_data = data.frame(k = 1:10, explained_var)

ggplot(elbow_data, 
       aes(x = k,  
           y = explained_var)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1) +            #<- sets the thickness of the line
  xlab('k') + 
  ylab('Inter-cluster Variance / Total Variance') + 
  theme_light()
#elbow plot says 2 centers is best!

#kmeans
set.seed(1)
kmeans_obj = kmeans(clust_df1, centers = 2, 
                    algorithm = "Lloyd")
kmeans_obj


won_clusters <- as.factor(kmeans_obj$cluster)

ggplot(df1, aes(x = num_turns, 
                y = on_play,
                color = won,  #<- tell R how to color 
                #   the data points
                shape = won_clusters)) + 
  geom_point(size = 6) +
  ggtitle("on_turn vs num_turns") +
  xlab("num_turns") +
  ylab("on_play") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2"),
                     values = c("1", "2")) +
  theme_light()

#The graph shows clear trends -- All of the 1s are together and all of the 2s are together! However, this could be because I chose a binary variable (on_play).

# Part 3: Review
#I feel like I have a better understanding of clustering after completing this activity.  While this data (or at least the variables I chose to work with) may not have been the best, I am able to better interpret the plot and I got to practice my data preparation!
  
  
  
