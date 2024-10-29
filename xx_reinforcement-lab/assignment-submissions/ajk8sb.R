library(tidyverse)

# read in data and create dataframe (df1)
df <- read_csv("data-summary.csv")
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

clustData <- df1[-c(1,2,5)]
clustData$on_play <- as.factor(as.numeric(clustData$on_play))

set.seed(42)
kmeansObj = kmeans(clustData, centers = 5, algorithm = 'Lloyd')

kmeansObj

clusters = as.factor(kmeansObj$cluster)

ggplot(clustData, aes(x = cora,
                      y = corc,
                      color = on_play,
                      shape = clusters)) + 
       geom_point(size = 6) +
       ggtitle('Mystery Data Plot') +
       xlab('cora') +
       ylab('corc') +
       scale_shape_manual(name = 'Cluster', 
                          labels = c('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'),
                          values = c('1', '2', '3', '4', '5')) +
       theme_light()