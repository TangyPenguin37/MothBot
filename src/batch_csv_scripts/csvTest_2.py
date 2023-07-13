import os
import csv
from tqdm import tqdm

csvOrig = "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data\\labelled\\data_normalized_onehot_simplified_corrected.csv"
csvNew = "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data\\info.wendy.batch1.batch2.sam.csv"

with open(csvOrig, 'r') as csvOrigFile:
    csvOrigReader = csv.reader(csvOrigFile, delimiter=',')
    csvOrigRows = [row for row in csvOrigReader]
    csvOrigFiles = [row[0] for row in csvOrigRows]
    csvOrigSpecies = [row[-1] for row in csvOrigRows]

    # remove file extension
    csvOrigFiles = [os.path.splitext(file)[0] for file in csvOrigFiles]

    # remove _d from the file names
    csvOrigFiles = [file[:-2] for file in csvOrigFiles]

    with open(csvNew, 'r') as csvNewFile:
        csvNewReader = csv.reader(csvNewFile, delimiter=',')
        csvNewRows = [row for row in csvNewReader]
        csvNewFiles = [row[0] for row in csvNewRows]
        csvNewSpecies = [row[1] for row in csvNewRows]

        for file in tqdm(csvOrigFiles):
            if file not in csvNewFiles:
                print(file)
            else:
                origIndex = csvOrigFiles.index(file)
                newIndex = csvNewFiles.index(file)
                if not ((csvOrigSpecies[origIndex] == "1"
                         and csvNewSpecies[newIndex] == "zea") or
                        (csvOrigSpecies[origIndex] == "0"
                         and csvNewSpecies[newIndex] == "arm")):
                    print(file)
                    print(csvOrigSpecies[origIndex])
                    print(csvNewSpecies[newIndex])
                    print()
