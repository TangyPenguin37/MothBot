import matplotlib.pyplot as plt
import numpy as np

# plot example data
csvData = np.loadtxt('data/unlabelled_batch1/data_formatted.csv',
                     delimiter=',',
                     skiprows=1,
                     usecols=1)

print(csvData.shape)

fig = plt.scatter(range(315), sorted(csvData))

plt.show()