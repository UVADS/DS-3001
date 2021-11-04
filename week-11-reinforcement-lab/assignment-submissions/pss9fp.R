library(tidyverse)
library(plotly)
library(htmltools)
library(devtools)
library(caret)
library(NbClust)

# read in data and create dataframe (df1)
df1 <- read_csv("data-frame.csv")

df1$main_colors <- fct_collapse(df1$main_colors,
                                Other = c("BG","BRG","U","UBG","UBR","UR","URG","W","WBG","WUB","WUR")
)


# make scatter plot comparing new features
ggplot(df1,aes(x=cora,y=corc, color=main_colors))+geom_point(aes(size=num_turns))

table(df1["main_colors"]) #all the oclors

kmeans = kmeans(df1[,c("num_turns","cora","corc")], centers = 2, 
                algorithm = "Lloyd")   #<- there are several ways of implementing k-means, see the help menu for a full list

clusters = as.factor(kmeans$cluster)
View(clusters)


ggplot(df1, aes(x = cora, 
                y = corc,
                color = clusters)) + 
  geom_point(aes(size=num_turns)) +
  ggtitle("Cora vs Corc") +
  xlab("Cora") +
  ylab("Corc") +
  
  theme_light()