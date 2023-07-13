library(colordistance)

files <- list.files("training_data/split", recursive = TRUE, full.names = TRUE)
files <- files[grepl("compressed", files)]

lower <- c(0, 0, 0)
upper <- c(0, 0, 0)

for (file in files) {
    test_hist <- colordistance::getImageHist(file)
    bg <- test_hist[c(4, 5, 8, 9, 18), ]

    lower <- lower + c(min(bg$r), min(bg$g), min(bg$b))
    upper <- upper + c(max(bg$r), max(bg$g), max(bg$b))
}

lower <- lower / length(files)
upper <- upper / length(files)

print(lower)
print(upper)
