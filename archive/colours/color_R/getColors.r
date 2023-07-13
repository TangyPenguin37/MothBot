library(colordistance)

work_with_test_file <- TRUE

print("Getting files...")

# get all files in parent directory
files <- list.files(
    path = "data/split_new",
    pattern = ".jpg",
    full.names = TRUE,
    recursive = TRUE
)

# files <- files[grepl("compressed", files)]

# print(files)

arm_files <- files[grepl("arm", files)]
zea_files <- files[grepl("zea", files)]

# stop("stop here")

if (work_with_test_file) {
    test_img <- files[1]
    test_hist <- colordistance::getImageHist(test_img)
    bg <- test_hist[c(4, 5, 8, 9, 18), ]

    lower <- c(min(bg$r) - 0.05, min(bg$g) - 0.1, min(bg$b) - 0.05)
    upper <- c(max(bg$r) + 0.05, max(bg$g) + 0.1, max(bg$b) + 0.05)

    filtered_img <- colordistance::getImageHist(
        test_img,
        lower = lower,
        upper = upper
    )

    colordistance::plotPixels(test_img, lower = lower, upper = upper)

    print("plotting...")

    # test_kmeans <- colordistance::getKMeanColors(
    #     test_img,
    #     n = 5,
    #     lower = lower,
    #     upper = upper,
    # )

    # stop("stop here")
}

lower <- c(0.3054571, 0.3994333, 0.1170031)
upper <- c(0.7407938, 0.8799308, 0.3893060)

kmeans_arm <- colordistance::getKMeansList(
    arm_files,
    bins = 4,
    lower = lower,
    upper = upper,
)

print(head(kmeans_arm, 3))

kmeans_clusters_arm <- colordistance::extractClusters(kmeans_arm)
print(head(kmeans_clusters_arm, 3))
plotClustersMulti(kmeans_clusters_arm)

kmeans_zea <- colordistance::getKMeansList(
    zea_files,
    bins = 4,
    lower = lower,
    upper = upper,
)

kmeans_clusters_zea <- colordistance::extractClusters(kmeans_zea)
print(head(kmeans_clusters_zea, 3))
plotClustersMulti(kmeans_clusters_zea)

# hist <- colordistance::getHistList(
#     arm_files,
#     bins = 2,
#     lower = lower,
#     upper = upper
# )

# colordistance::plotClustersMulti(hist, title = "Histogram method")
