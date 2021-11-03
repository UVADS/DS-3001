## Metrics Evaluation Lab

Throughout your early career as a Data Scientist you've built article summaries, 
explored NBA talent, and analyzed text on climate change news but you've suddenly realized you need to enhance your ability to assess the models you are building. 
As the most important part about understanding any machine learning model 
(or any model, really) is understanding it's weakness and/or vulnerabilities. 

In doing so you've decided to practice on datasets that are of interest to you, 
and use a approach to which you are (becoming) familiar, kNN. 

Part 1. Select either as a lab group or individuals a dataset that is of interest to you/group. Define a question that can be answered using classification, specifically kNN, for the dataset. 

Part 2. In consideration of all the metrics we discussed select a few (threeish) key metrics that should be tracked given the question you are working to solve. 

Part 3. Build a kNN model and evaluate the model using using the metrics discussed in class (Accuracy, TPR, FPR, F1, Kappa, LogLoss and ROC/AUC). Make sure to calculate the baserate or prevalence to provide a reference for some of these measures. Even though you are generating many of the metrics we discussed summarize the output of the key metrics you established in part 2. 

Part 4. Consider where miss-classification errors are occurring, is there a pattern? If so discuss this pattern and why you think this is the case. 

Part 5. Based on your exploration in Part 3/4, change the threshold using the function provided (in the in-class example), what differences do you see in the evaluation metrics? Speak specifically to the metrics that are best suited to address the question you are trying to answer. 

Part 6. Summarize your findings speaking through your question, what does the evaluation outputs mean when answering the question? Also, make recommendations on improvements. 

Recommendations for improvement might include gathering more data, adjusting the threshold, adding new features, changing your questions or maybe that it's working fine at the current level and nothing should be done. 

Submit a .Rmd file along with the data used or access to the data sources to the Collab site. You can work together with your groups but submit individually. 

Keys to Success: 
* Thoughtfully creating a question that aligns with your dataset and classification 
* Using the evaluation metrics correctly - some require continuous probability outcomes (LogLoss) while others require binary predictions (pretty much all the rest).
* Evaluation is not about the metrics per say, but what they mean, speaking through your question in light of the evaluation metrics is the primary objective of this lab. Think of yourself as a "model detective" that works to leave no stone unturned!  
