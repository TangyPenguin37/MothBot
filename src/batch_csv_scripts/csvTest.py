import os
import csv
from tqdm import tqdm

csvAll = "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data\\all.csv"
csvInfo = "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data\\info.wendy.batch1.batch2.sam.csv"

folders = [
    "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data\\batch1\\orig\\jpg",
    "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data\\batch2\\jpg"
]

# Get all the files in the folders
files = [os.listdir(folder) for folder in folders]

# Flatten the list
files = [item for sublist in files for item in sublist]

# Get the file names
files = [os.path.splitext(file)[0] for file in files]

# Get the unique file names
files = list(set(files))

# remove _d from the file names
files = [file[:-2] for file in files]

# Get the csv file
with open(csvAll, 'r') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    # get first column
    csvRows = [row for row in csvReader]
    csvFiles = [row[0] for row in csvRows]

    for file in tqdm(files):
        if file in csvFiles:
            index = csvFiles.index(file)
            if csvRows[index][1] == "Mastersheet":
                with open(csvInfo, 'r') as csvInfoFile:
                    csvInfoReader = csv.reader(csvInfoFile, delimiter=',')
                    csvInfoRows = [row for row in csvInfoReader]
                    csvInfoFiles = [row[0] for row in csvInfoRows]
                    if file not in csvInfoFiles:
                        print(file)
        else:
            print(file)
