import os
import numpy as np
import matplotlib.pyplot as plt

directory = os.path.dirname(os.path.realpath(__file__))

with open(directory + '/features_partial.txt', encoding='UTF8') as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]

    values = {}

    lines = list(
        filter(
            lambda x: "Wing" not in x and x != "" and "efd" not in x and
            "percentage" not in x, lines))

    for line in lines:
        line = line.split(" ")
        key = line[0][:-1]
        value = float(line[1])

        if key not in values:
            values[key] = []

        values[key].append(value)

    # sort by mean
    values = dict(
        sorted(values.items(), key=lambda item: np.mean(item[1]),
               reverse=True))

    for value in values.values():
        assert len(value) == 4

    # plot as bar chart
    plt.bar(np.arange(len(values)), [value[0] for value in values.values()],
            0.2)
    plt.bar(
        np.arange(len(values)) + 0.2, [value[1] for value in values.values()],
        0.2)
    plt.bar(
        np.arange(len(values)) + 0.4, [value[2] for value in values.values()],
        0.2)
    plt.bar(
        np.arange(len(values)) + 0.6, [value[3] for value in values.values()],
        0.2)

    plt.xticks(np.arange(len(values)) + 0.3, values.keys(), rotation=90)
    plt.legend(["front left", "front right", "back left", "back right"])
    plt.xlabel("Feature")
    plt.ylabel("Percentage")
    plt.title("Percentage of Features in Graphs")
    plt.tight_layout()

    plt.show()