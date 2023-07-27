import os
import csv

root = "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data"

infoCSV = os.path.join(root, "all_info.csv")
dataCSV = os.path.join(root, "batch2", "data_all.csv")
newCSV = os.path.join(root, "batch2", "data_all_w_species.csv")

with open(infoCSV, "r", encoding='UTF8') as info_file:
    info_reader = csv.reader(info_file, delimiter=",")
    info = list(info_reader)

with open(dataCSV, "r", encoding='UTF8') as data_file:
    data_reader = csv.reader(data_file, delimiter=",")
    data = list(data_reader)

with open(newCSV, "w", encoding='UTF8', newline="") as new_file:
    new_writer = csv.writer(new_file, delimiter=",")
    new_writer.writerow(data[0] + ["species"])
    for row in data[1:]:
        filename = row[0][:-6]
        for info_row in info:
            if info_row[0] == filename:
                new_writer.writerow(row + [info_row[1]])
                break
        else:
            new_writer.writerow(row + [""])