# Week 11: Reinforcement lab
Welcome to week 11 where we will do a reinforcement lab.

Let's take a look at the standard diagram (credit ref. [1])

![](https://github.com/alonzi/DS-3001/blob/a41a636f4cba7ad7869342056b8f3e90dc4507d8/week-11-reinforcement-lab/reinforcement.jpeg)

Ok, that was a dad joke, I couldn't help myself. That diagram is about Reinforcement Learning. Let's get to what we really mean, reinforcing the concepts from the past few weeks. The way we are going to do that is review, question, explore. (also known as KWL ref. [2]).

## A little about Pete
1. PhD in particle physics
2. Lead of Research Computing for the School of Data Science
3. LPA2A@virginia.edu

## Goals for Today
* Reinforce your understanding of previous topics 
* Everyone makes a PR to this repo (details at end, graded on completion)

# Part 1: Review
The syllabus shows the past few weeks have focused on
* Unsupervised KNN
* Evalucation metrics

## Exercise 1: What do you know?
For this exercise we will work in your teams.

1. There are 10 minutes on the clock
2. Discuss what you know and note those things
3. If you bump into something you know now write that down too in a different spot.

All together: Tell the room what you all wrote down

## Exercise 2: What do you want to know?
You get the drill, 10 mintues are on the clock.

1. Discuss with your team what you want to know. Where are you shaky? Where would you like more time to practice?
2. N.B.: This is tricky, you need to put yourself out there, trust your partners, be vulnerable.

All together: Tell the room what you want to know, what questions do you want to answer, where do you want to reinforce?

# Part 2: Explore
In part two we are going to rip into some data and explore. The overall goal is to reinforce what you started to learn in the past few weeks. Keep your questions in mind as we go, let them guide you. If you want to take a scenic route because it will help to reinforce, do it. Remember the goal is reinforcement, not completion of some assignment (you can always finish it for homework).


## Today's Dataset
* We will be using a mystery dataset today: [mystery data](https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game-data.MID.PremierDraft.tar.gz)
* Courtesy of 17 lands, ref [3]
* N.B.: if you look at the url you will note it comes from s3.amazonaws.com

## Let's Get Started
1. Download the data (hint: click the link above, it should open a download)
2. Un zip the data (hint: `tar -xzf game-data.MID.PremierDraft.tar.gz`)
3. Make an abbreviated data file (hint: `head -n 1000 game_data_public.MID.PremierDraft.csv > data-summary.csv`)
4. Load the data into a dataframe (hint: see scratch.R)
5. Select relevant columns: main_colors,opp_colors,on_play,num_turns,won (hint: see scratch.R) 
4. **Hard Step** compute two correlation metrics (hint: see scratch.R)
5. Make scatter plot of the two correlation metrics (hint: see scratch.R)

(tar,knn,why cluster, save 20 for show and tell)

## Let's challenge ourselves
Now explore, get creative, answer your questions, reinforce what you have learned.


# Part 3: Reflect


# PR details
1. fork this repo: https://github.com/alonzi/DS-3001
2. create a file called [uvaNetID].R in the folder petes-session (eg for me: petes-session/LPA2A.R) << this may go to knitr markdwon something >>
3. the file must contain one thing you haven't done before aka something you learned today in this session
4. submit your pull request (don't forget to explain it in the PR itself)
5. stick with it, your assignment isn't complete until I merge the PR, we will have discussions in the comments
  
  
# Footnotes
1. https://www.kdnuggets.com/2018/03/5-things-reinforcement-learning.html
2. https://en.wikipedia.org/wiki/KWL_table#:~:text=The%20letters%20KWL%20are%20an,methods%20of%20teaching%20and%20learning.
3. https://www.17lands.com/public_datasets
