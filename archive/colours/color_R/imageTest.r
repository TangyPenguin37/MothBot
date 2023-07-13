library(colordistance)

test_file <- "data/split_new/zea_leftfront/CAM046043_d.jpg"

img <- colordistance::loadImage(test_file)

# print(names(img))

matrix <- img$filtered.rgb.2d

# # transpose
matrix <- t(matrix) * 255

hsv <- rgb2hsv(matrix)

print(hsv)

# print(matrix)