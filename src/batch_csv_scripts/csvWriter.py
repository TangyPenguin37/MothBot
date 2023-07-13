import os
import csv

locations = ["front", "rear"]
sides = ["left", "right"]

root = "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data"

batch1data = os.path.join(root, "batch1", "data_new.csv")

# for file in batch2 (by wing)
# add details from imagej data (all blank if empty) then efd then colour and copy to new file