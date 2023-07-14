import os
import random
import csv

# This script is used to set aside a portion of the data for testing purposes.
# The script will take 10% of the data for each class and put it into a separate file.

def splitData():
    # Get the current working directory
    cwd = os.getcwd()

    # Get the path to the data folder
    rootDataPath = os.path.join(cwd, "data")
    csvDataPath = os.path.join(rootDataPath, "data_formatted.csv")

    with open(csvDataPath, newline='', encoding='UTF8') as csvfile:
        reader = csv.reader(csvfile)

        header = next(reader)

        # Create a dictionary to store the data
        armIDs = set()
        zeaIDs = set()
        hybIDs = set()

        reader = list(reader)

        # Iterate through the rows of the csv file
        for row in reader:
            # Get the class of the current row
            currentClass = row[-1]

            # Get the file name of the current row
            currentID = row[0][:-6]

            match currentClass:
                case "0":
                    armIDs.add(currentID)
                case "0.5":
                    hybIDs.add(currentID)
                case "1":
                    zeaIDs.add(currentID)

        armIDs = sorted(armIDs)
        hybIDs = sorted(hybIDs)
        zeaIDs = sorted(zeaIDs)

        # randomly select 10% of the data for each class
        armTestIDs = random.sample(armIDs, int(len(armIDs) * 0.1))
        hybTestIDs = random.sample(hybIDs, int(len(hybIDs) * 0.1))
        zeaTestIDs = random.sample(zeaIDs, int(len(zeaIDs) * 0.1))

        testIDs = armTestIDs + hybTestIDs + zeaTestIDs

        with open("data\\split_data\\test_samples.csv",
                  "w", newline='',
                  encoding='UTF8') as testFile, open("data\\split_data\\train_samples.csv",
                                                                 "w", newline='',
                                                                 encoding='UTF8') as trainFile:

            testWriter = csv.writer(testFile)
            trainWriter = csv.writer(trainFile)

            testWriter.writerow(header)
            trainWriter.writerow(header)

            for row in reader:
                # Get the file name of the current row
                currentID = row[0][:-6]

                if currentID in testIDs:
                    testWriter.writerow(row)
                else:
                    trainWriter.writerow(row)
        
def main():
    splitData()

if __name__ == "__main__":
    main()