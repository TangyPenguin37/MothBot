import os
import csv

DESIRED = "train"

rootFilepath = os.path.join(os.path.dirname(__file__), "../..")

trainDataFilepath = os.path.join(rootFilepath, "data", "split_data", "train")
testDataFilepath = os.path.join(rootFilepath, "data", "split_data", "test")

trainCSVFilepath = os.path.join(trainDataFilepath,
                                "train_data_formatted_all.csv")
testCSVFilepath = os.path.join(testDataFilepath, "test_data_formatted.csv")

imagesFilepath = os.path.join(trainDataFilepath, "images")

def getTrainIDs():
    with open(trainCSVFilepath, "r", encoding='UTF8') as trainCSV:
        trainReader = csv.reader(trainCSV)
        next(trainReader)  # skip header
        return set(row[0][:-6] for row in trainReader)

def getTestIDs():
    with open(testCSVFilepath, "r", encoding='UTF8') as testCSV:
        testReader = csv.reader(testCSV)
        next(testReader)  # skip header
        return set(row[0][:-6] for row in testReader)

testIDs = getTestIDs()
trainIDs = getTrainIDs()

assert len(testIDs.intersection(trainIDs)) == 0

files = os.walk(imagesFilepath)

for root, _, filenames in files:
    for filename in filenames:
        if filename.endswith(".JPG"):
            filepath = os.path.join(root, filename)
            if DESIRED == "test":
                if filename[:-6] not in testIDs:
                    os.remove(filepath)
            elif DESIRED == "train":
                if filename[:-6] not in trainIDs:
                    os.remove(filepath)
            else:
                raise ValueError("DESIRED must be 'test' or 'train'")