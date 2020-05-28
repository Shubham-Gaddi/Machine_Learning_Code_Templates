# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Additional Preprocessing
dataset = pd.read_csv(' ', header = None)
sample = []
for i in range(0, n):
    sample.append([str(dataset.values[i,j]) for j in range(0, w)])

# Training The Apriori Model
# WARNING: Make sure to place the apyori.py file intp the working directory before running the code
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Checking Results
results = list(rules)  
print(results) 