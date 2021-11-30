# herg
Contains analyses for hERG classification paper

A colab notebook is at https://colab.research.google.com/drive/1g9JlkIxmejK-xgfVR7zeY9xr4BwrsjiA#scrollTo=2F8-oSXeopMa

feature_sets/ # contains feature data sets 

dataUtil.py # for conditional probabilities 
mlUtil.py # for ML strategies 


# TODOS
- something is fishy about the zscores, so I want to look into that later
- Splitting the dataset into 70% for training, 30% for testing leads to very noisy results for the test set. Most likely, in order to this properly, I will probably need to bootstrap. For now I'm using the entire data set for training and testing
