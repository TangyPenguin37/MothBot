import os
import csv

dataFolder = os.path.join(os.getcwd(), 'data', 'split_data', 'train', 'without_hybrids')

oldCSVFilepath = os.path.join(dataFolder, 'new_results_no_hybrids_grouped.csv')
newCSVFilepath = os.path.join(dataFolder, 'new_results_no_hybrids_grouped_further.csv')

with open(oldCSVFilepath, 'r', encoding='UTF8') as oldCSVFile:
    with open(newCSVFilepath, 'w', encoding='UTF8', newline="") as newCSVFile:
        rows = list(csv.reader(oldCSVFile, delimiter=','))
        writer = csv.writer(newCSVFile, delimiter=',')

        # remove header row
        headers = rows.pop(0)
        headers = headers[3:-1]
        print(headers)

        newHeaders = ['filename'
                      ] + [f'{i}_{j}' for j in range(4)
                           for i in headers] + ['species']

        writer.writerow(newHeaders)

        IDs = [set() for _ in range(4)]

        for row in rows:

            index = 2 * int(row[1]) + int(row[2])

            IDs[index].add(row[0])

        commonIDs = IDs[0].intersection(IDs[1], IDs[2], IDs[3])

        print(len(commonIDs))

        newRows = {ID: [ID] for ID in sorted(commonIDs)}

        for row in rows:
            ID = row[0]

            if ID in commonIDs:
                newRows[ID].extend(row[3:-1])
                if row[1] == '1' and row[2] == '1':
                    newRows[ID].extend(row[-1])

        writer.writerows(newRows.values())