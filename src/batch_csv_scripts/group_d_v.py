import os
import csv

dataFolder = os.path.join(os.getcwd(), 'data', 'split_data', 'train', 'without_hybrids')

oldCSVFilepath = os.path.join(dataFolder, 'new_results_no_hybrids.csv')
newCSVFilepath = os.path.join(dataFolder, 'new_results_no_hybrids_grouped.csv')

with open(oldCSVFilepath, 'r', encoding='UTF8') as oldCSVFile:
    with open(newCSVFilepath, 'w', encoding='UTF8', newline="") as newCSVFile:
        rows = list(csv.reader(oldCSVFile, delimiter=','))
        writer = csv.writer(newCSVFile, delimiter=',')

        # remove header row
        headers = rows.pop(0)
        headers = headers[4:-1]
        print(headers)

        newHeaders = ['filename', 'location', 'side'
                      ] + [f'{i}_{j}' for j in ['d', 'v']
                           for i in headers] + ['species']

        writer.writerow(newHeaders)

        for row_d in rows:
            filename = row_d[0]
            ID = filename.split('_')[0]

            if '_v' in filename:
                continue

            data_d = row_d[4:-1]

            data_v = []

            for row_v in rows:
                if ID in row_v[0] and '_v' in row_v[0] and row_v[1] == row_d[
                        1] and row_v[2] == row_d[2]:
                    data_v = row_v[4:]
                    break

            if data_v == []:
                print(f'Error: {ID} has no _v data')
                continue

            newData = [ID] + row_d[1:3] + data_d + data_v
            writer.writerow(newData)

        for row_v in rows:
            filename = row_v[0]
            ID = filename.split('_')[0]

            if '_d' in filename:
                continue

            dFound = False

            for row_d in rows:
                if ID in row_d[0] and '_d' in row_d[0] and row_d[1] == row_v[
                        1] and row_d[2] == row_v[2]:
                    dFound = True
                    break

            if not dFound:
                print(f'Error: {ID} has no _d data')
