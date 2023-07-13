import numpy as np
import matplotlib.pyplot as plt

COLOR = 0

csvData = np.loadtxt('data/labelled/data_normalized_onehot.csv',
                     delimiter=',',
                     skiprows=1,
                     usecols=range(1, 54))

labels = np.loadtxt('data/labelled/data_normalized_onehot.csv',
                    delimiter=',',
                    dtype=np.str0,
                    max_rows=1,
                    usecols=range(1, 54))

columns = [x for x in range(4, 53) if f"colour_{COLOR}" in labels[x]]
columns = sorted([0, 1, 2, 3, 52] + columns)

# print(columns)

front = csvData[csvData[:, 2] == 1]
rear = csvData[csvData[:, 3] == 1]

front = front[:, columns]
rear = rear[:, columns]

left_front = front[front[:, 0] == 1]
right_front = front[front[:, 1] == 1]
left_rear = rear[rear[:, 0] == 1]
right_rear = rear[rear[:, 1] == 1]

# print(left_front[:, 4:])

# plot on 3d scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatterColours_leftfront = ['r' if x == 1 else 'b' for x in left_front[:, -1]]
scatterColours_rightfront = [
    'r' if x == 1 else 'b' for x in right_front[:, -1]
]
scatterColours_leftrear = ['r' if x == 1 else 'b' for x in left_rear[:, -1]]
scatterColours_rightrear = ['r' if x == 1 else 'b' for x in right_rear[:, -1]]

ax.scatter(left_front[:, 4],
           left_front[:, 5],
           left_front[:, 6],
           c=scatterColours_leftfront,
           marker='o')

ax.scatter(right_front[:, 4],
           right_front[:, 5],
           right_front[:, 6],
           c=scatterColours_rightfront,
           marker='o')

# ax.scatter(left_rear[:, 4],
#            left_rear[:, 5],
#            left_rear[:, 6],
#            c=scatterColours_leftrear,
#            marker='o')

# ax.scatter(right_rear[:, 4],
#            right_rear[:, 5],
#            right_rear[:, 6],
#            c=scatterColours_rightrear,
#            marker='o')

ax.set_xlabel(labels[columns[4]])
ax.set_ylabel(labels[columns[5]])
ax.set_zlabel(labels[columns[6]])

plt.show()