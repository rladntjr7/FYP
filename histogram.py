import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the csv file
df = pd.read_csv('_annotations.csv').to_numpy()
names = df[:, 0]

# make a list of unique names
unique_names = []
for name in names:
    if name not in unique_names:
        unique_names.append(name)

# make a list of unique names and their counts
unique_names_counts = []
for name in unique_names:
    unique_names_counts.append(len(list(name_ for name_ in names if name_ == name)))

unique_names_counts.append(0)
# make a histogram
plt.hist(unique_names_counts, bins=range(0, 48, 1))
plt.xlabel('Sperm Count')
plt.ylabel('Frequency')
plt.show()
