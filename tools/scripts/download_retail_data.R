# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script retrieves the orangeJuice dataset from the bayesm R package and saves the data as csv

# Load the data
library(bayesm)
data("orangeJuice")
yx <- orangeJuice[[1]]
storedemo <- orangeJuice[[2]]

# Create a data directory
path_to_data <- "~/ojdata"
fpath <- file.path(path_to_data)
if(!dir.exists(fpath)) dir.create(fpath)

# Write the data to csv files
write.csv(yx, file = file.path(fpath, "yx.csv"), quote = FALSE, na = " ", row.names = FALSE)
write.csv(storedemo, file = file.path(fpath, "storedemo.csv"), quote = FALSE, na = " ", row.names = FALSE)

print(paste("Data download completed. Data saved to ", path_to_data))
