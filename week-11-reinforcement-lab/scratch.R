
library(tidyverse)

df <- read_csv("data-summary.csv")

df1 <- select(df,main_colors,opp_colors,on_play,num_turns,won,"deck_Adeline, Resplendent Cathar":"deck_Wrenn and Seven")

df2 <- mutate(df1,comp = c("deck_Adeline, Resplendent Cathar","deck_Wrenn and Seven"))


df1[,c("deck_Adeline, Resplendent Cathar","deck_Wrenn and Seven")]


# as.vector(as.matrix(df[,c("alpha", "gamma", "zeta")]))
