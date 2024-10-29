## Metrics Evaluation Lab

Throughout your early career as a Data Scientist you've spent most your time cleaning data, but now you are starting to build models and have come to realize the most important part about understanding any machine learning model (or any model, really) is understanding it's weakness and vulnerabilities. 

In doing so you've decided to practice on a dataset about mushrooms, because after all if you don't know how to evaluate a model thoroughly you'll be in real truffle (ha...ha) and use a approach to which you are familiar, kNN. 

Part 1. Using the [mushroom dataset](https://archive.ics.uci.edu/static/public/848/secondary+mushroom+dataset.zip), Define a question that can be answered using classification, specifically kNN.

 - [Mushroom Documentation](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)

Part 2. Build a kNN model and evaluate the model using using the metrics discussed in class (Accuracy, TPR, FPR, F1, Kappa, LogLoss and ROC/AUC). Make sure to calculate the prevalence to provide a reference for some of these measures. Make sure to properly clean and prepare the data ahead of building the model.  

Part 3. In consideration of all the metrics you just used are there a few that seem more important given the question you are asking? 

Part 4. Consider where miss-classification errors are occurring, is there a pattern? If so discuss this pattern and why you think this is the case. 

Part 5. Based on your exploration in Part 3/4, change the threshold using the function provided (in the in-class example), what differences do you see in the confusion metric? Does it get better at addressing your question or not, why?  

Part 6. Use a metric we did not discuss in class (reference the sklearn model metrics documentation). Once you have the output, summarize in a sentence or two what the metric is and what it means in the context of your question.

Part 7. Summarize your findings speaking through your question, what does the evaluation outputs mean when answering the question? Also, make recommendations on improvements. 

Recommendations for improvement might include gathering more data, adjusting the threshold, adding new features, changing your questions or maybe that it's working fine at the current level and nothing should be done. 

Submit a .py or ipynb file along with the data used or access to the data sources to the Canvas site. You can work together with your groups but submit individually. 

Keys to Success: 

- Thoughtfully creating a question that aligns with the dataset and classification

- Using the evaluation metrics correctly - we are focusing on classification not regression

- Evaluation is not about the metrics per say, but what they mean, speaking through your question in light of the evaluation metrics is the primary objective of this lab. Think of yourself as a "model detective" that works to leave no stone unturned! 

- Remember, be patience and double check your code or you might find yourself in real shiitake, :)
