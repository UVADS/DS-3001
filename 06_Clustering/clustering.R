#============================================================
##### CLUSTERING: FINDING PATTERNS IN YOUR DATA ####
#============================================================

#### Slide 21: Step 1: load packages and data ####

# Install packages.
install.packages("e1071")     #<- package used to run clustering analysis

# Load libraries.
library(e1071)
library(tidyverse)
library(plotly)
library(htmltools)
library(devtools)

#library(installr)



library(help = "e1071")#<- learn about all the functionality of the package,
#   be well informed about what you're doing

#==================================================================================

#### Slide 22: Step 1: load packages and data ####

house_votes_Dem = read_csv("data/house_votes_Dem.csv")

# What does the data look like?
View(house_votes_Dem)
str(house_votes_Dem)
table(house_votes_Dem$party.labels)

house_votes_Rep = read_csv("data/house_votes_Rep.csv")

table(house_votes_Rep$party.labels)
View(house_votes_Rep)
#==================================================================================

#### Slide 23: Step 2: run k-means ####

# Define the columns to be clustered by sub-setting the data.
# Placing the vector of columns after the comma inside the 
# brackets tells R that you are selecting columns.
clust_data_Dem = house_votes_Dem[, c("aye", "nay", "other")]
View(clust_data_Dem)

# Run an algorithm with 2 centers.
# kmeans uses a different starting data point each time it runs.
# Make the results reproducible with the set.seed() function.
set.seed(1)
kmeans_obj_Dem = kmeans(clust_data_Dem, centers = 2, 
                        algorithm = "Lloyd")   #<- there are several ways of implementing k-means, see the help menu for a full list

# What did the kmeans function produce, 
# what does the new variable kmeans_obj contain?
kmeans_obj_Dem

# View the results of each output of the kmeans function.
head(kmeans_obj_Dem)


#==================================================================================

#### Slide 28: Step 3: visualize plot ####

# Tell R to read the cluster labels as factors so that ggplot2 
# (the graphing package) can read them as category labels instead of 
# continuous variables (numeric variables).
party_clusters_Dem = as.factor(kmeans_obj_Dem$cluster)

# What does the kmeans_obj look like?
View(party_clusters_Dem)

# Set up labels for our data so that we can compare Democrats and Republicans.
party_labels_Dem = house_votes_Dem$party.labels
View(party_labels_Dem)

#==================================================================================

#### Slide 29: Step 3: visualize plot ####

View(house_votes_Dem)
View(party_clusters_Dem)

ggplot(house_votes_Dem, aes(x = aye, 
                            y = nay,
                            shape = party_clusters_Dem)) + 
  geom_point(size = 6) +
  ggtitle("Aye vs. Nay votes for Democrat-introduced bills") +
  xlab("Number of Aye Votes") +
  ylab("Number of Nay Votes") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2"),
                     values = c("1", "2")) +
  theme_light()

#==================================================================================

#### Slide 32: Step 5: validate results ####

ggplot(house_votes_Dem, aes(x = aye, 
                            y = nay,
                            color = party_labels_Dem,  #<- tell R how to color 
                            #   the data points
                            shape = party_clusters_Dem)) + 
  geom_point(size = 6) +
  ggtitle("Aye vs. Nay votes for Democrat-introduced bills") +
  xlab("Number of Aye Votes") +
  ylab("Number of Nay Votes") +
  scale_shape_manual(name = "Cluster", 
                     labels = c("Cluster 1", "Cluster 2"),
                     values = c("1", "2")) +
  scale_color_manual(name = "Party",         #<- tell R which colors to use and
                     #   which labels to include in the legend
                     labels = c("Republican", "Democratic"),
                     values = c("red", "blue")) +
  theme_light()


# Save your graph. For Windows, use setwd("C:/file path")

ggsave("US House Votes for Dem Bills.png", 
       width = 10, 
       height = 5.62, 
       units = "in")
#==================================================================================

#### Slide 35: Clustering vs. visualizing ####

# We can visualize votes in 3D with the following code.
# View our data.
View(house_votes_Dem)

# Assign colors by party in a new data frame.
party_color3D_Dem = data.frame(party.labels = c("Democrat", "Republican"),
                               color = c("blue", "red"))

View(party_color3D_Dem)


# Join the new data frame to our house_votes_Dem data set.
house_votes_color_Dem = inner_join(house_votes_Dem, party_color3D_Dem)

View(house_votes_color_Dem)

house_votes_color_Dem$Last.Name <- gsub("[^[:alnum:]]", "", house_votes_color_Dem$Last.Name)

# Use plotly to do a 3d imaging 

fig <- plot_ly(house_votes_color_Dem,type = "scatter3d",
               mode="markers", 
               x = ~aye, 
               y = ~nay, 
               z = ~other, 
               color = ~color, 
               colors = c('#0C4B8E','#BF382A'), 
               text = ~paste('Representative:',Last.Name))


fig
dev.off()

#==================================================================================

#### Slide 41: How good is the clustering? ####

# Inter-cluster variance,
# "betweenss" is the sum of the distances between points 
# from different clusters.
num_Dem = kmeans_obj_Dem$betweenss

# Total variance, "totss" is the sum of the distances
# between all the points in the data set.
denom_Dem = kmeans_obj_Dem$totss

# Variance accounted for by clusters.
(var_exp_Dem = num_Dem / denom_Dem)


#==================================================================================

#### Slide 43: Elbow method: measure variance ####

# Run an algorithm with 3 centers.
set.seed(1)
kmeans_obj_Dem = kmeans(clust_data_Dem, centers = 3, algorithm = "Lloyd")

# Inter-cluster variance.
num_Dem3 = kmeans_obj_Dem$betweenss

# Total variance.
denom_Dem3 = kmeans_obj_Dem$totss

# Variance accounted for by clusters.
(var_exp_Dem3 = num_Dem3 / denom_Dem3)


#==================================================================================

#### Slide 45: Automating a step we want to repeat ####

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

#==================================================================================

#### Slide 46: automating a step we want to repeat ####

# Recall the variable we are using for the data that we're clustering.
View(clust_data_Dem)

# The sapply() function plugs in several values into our explained_variance function.
#sapply() takes a vector, lapply() takes a dataframe
explained_var_Dem = sapply(1:10, explained_variance, data_in = clust_data_Dem)

View(explained_var_Dem)


# Data for ggplot2.
elbow_data_Dem = data.frame(k = 1:10, explained_var_Dem)
View(elbow_data_Dem)

#==================================================================================

#### Slide 47: Elbow method: plotting the graph ####

# Plotting data.
ggplot(elbow_data_Dem, 
       aes(x = k,  
           y = explained_var_Dem)) + 
  geom_point(size = 4) +           #<- sets the size of the data points
  geom_line(size = 1) +            #<- sets the thickness of the line
  xlab('k') + 
  ylab('Inter-cluster Variance / Total Variance') + 
  theme_light()

#==================================================================================

#### Slide 50: NbClust: k by majority vote ####

# Install packages.
#install.packages("NbClust") if needed
library(NbClust)

# Run NbClust.
(nbclust_obj_Dem = NbClust(data = clust_data_Dem, method = "kmeans"))

# View the output of NbClust.
nbclust_obj_Dem

# View the output that shows the number of clusters each method recommends.
View(nbclust_obj_Dem$Best.nc)

#==================================================================================

#### Slide 54: NbClust: k by majority vote ####

# Subset the 1st row from Best.nc and convert it 
# to a data frame so ggplot2 can plot it.
freq_k_Dem = nbclust_obj_Dem$Best.nc[1,]
freq_k_Dem = data.frame(freq_k_Dem)
View(freq_k_Dem)

# Check the maximum number of clusters suggested.
max(freq_k_Dem)

#essentially resets the plot viewer back to default
dev.off()

# Plot as a histogram.
ggplot(freq_k_Dem,
       aes(x = freq_k_Dem)) +
  geom_bar() +
  scale_x_continuous(breaks = seq(0, 15, by = 1)) +
  scale_y_continuous(breaks = seq(0, 12, by = 1)) +
  labs(x = "Number of Clusters",
       y = "Number of Votes",
       title = "Cluster Analysis")


#==================================================================================

#Now you try with the Republican Data and a NBA Stat Example 

#==================================================================================

