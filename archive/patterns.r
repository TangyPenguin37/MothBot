library(patternize)

# open txt file of coordinates

filepath <- "C:/Users/samue/OneDrive - University of Cambridge/Summer Research Project/data/split/arm/front/left/left.txt"

coords <- read.table(filepath)

# split into groups of 5

coords <- split(coords, rep(1:(nrow(coords) / 5), each = 5))

# print(coords)

image_IDs <- tools::file_path_sans_ext(dir("C:/Users/samue/OneDrive - University of Cambridge/Summer Research Project/data/split/arm/front/left/", ".jpg"))

print(image_IDs)

image_list <- makeList(image_IDs,
    type = "image",
    prepath = "C:/Users/samue/OneDrive - University of Cambridge/Summer Research Project/data/split/arm/front/left/",
    extension = ".jpg"
)

print(image_list)

aligned_images <- alignLan(image_list, coords, 5, cartoonID = "CAM046001_d")

print(aligned_images)