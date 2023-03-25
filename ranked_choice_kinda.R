install.packages("vote")
library(vote)

mascot <- read.csv("mascot_ranking.csv")

xx <- stv(mascot, nseats=1, verbose=TRUE)

View(mascot)
data()
install.packages("reshape2")
library(reshape2)
dat2b <- melt(mascot)
View(dat2b)

library(tidyverse)

#couldn't remember how to do this so had to look it up in the documentation
long <- pivot_wider(dat2b,id_cols = 2)

barplot(dat2b$value)

str(dat2b)

dat2b$value <- as.factor(dat2b$value)

dat2b %>% group_by(variable)

table=table(dat2b$variable,dat2b$value)



data_boom <- as.data.frame.matrix(table)

View(data_boom)

results <- stv(data_boom, nseats=1, verbose=TRUE)




