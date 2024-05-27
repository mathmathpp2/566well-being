rm(list=ls())

if (!requireNamespace("BiocManager", quietly = TRUE)){
  install.packages("BiocManager")}

if (!requireNamespace("Rgraphviz", quietly = TRUE))
  BiocManager::install("Rgraphviz",update=FALSE)
library("Rgraphviz")

if (!requireNamespace("RBGL", quietly = TRUE)){
  BiocManager::install("RBGL",update=FALSE)}
library("RBGL")

if (!requireNamespace("abind", quietly = TRUE)){
  install.packages("abind",update=FALSE)}
library("abind")

if (!requireNamespace("corpcor", quietly = TRUE)){
  install.packages("corpcor",update=FALSE)}
library("corpcor")

if (!requireNamespace("sfsmisc", quietly = TRUE)){
  install.packages("sfsmisc",update=FALSE)}
library("sfsmisc")

if (!requireNamespace("robustbase", quietly = TRUE)){
  install.packages("robustbase",update=FALSE)}    
library("robustbase")

if (!requireNamespace("pcalg", quietly = TRUE)){
  install.packages("pcalg",update=FALSE)}
library("pcalg")

if (!requireNamespace("graph", quietly = TRUE)){
  install.packages("graph",update=FALSE)}
library("graph")

data_full <- read.csv("features_dropna.csv")

subset <- c("P1", "P2", "P3", "P4", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "Z1", "Z2", "Z3", "Y4")
subset

data <- data_full[, subset]

dim(data)

cor(data)

pairs(data, lower.panel = NULL)


n <- nrow(data)
p <- ncol(data)
n
p
indepTest <- gaussCItest
suffStat <- list(C=cor(data), n = n)

## estimate CPDAG
# https://cran.r-project.org/web/packages/pcalg/pcalg.pdf

alpha <- 0.05

sink("output-pcfit.txt")

# pc.fit <- pc(suffStat, indepTest, p = p, alpha = alpha, verbose = TRUE, numCores = 8)
pc.fit <- pc(suffStat, indepTest, labels = subset, alpha = alpha, verbose = TRUE, numCores = 8)
# pc.fit <- pc(suffStat, indepTest, labels = subset, alpha = alpha, verbose = TRUE, numCores = 8, conservative = TRUE)
# pc.fit <- pc(suffStat, indepTest, labels = subset, alpha = alpha, verbose = TRUE, numCores = 8, maj.rule = TRUE)
# pc.fit <- pc(suffStat, indepTest, labels = subset, alpha = alpha, verbose = TRUE, numCores = 8, conservative = TRUE, solve.confl = TRUE)
# pc.fit <- pc(suffStat, indepTest, labels = subset, alpha = alpha, verbose = TRUE, numCores = 8, maj.rule = TRUE, solve.confl = TRUE)

sink()

showAmat(pc.fit)
showEdgeList(pc.fit, subset)

# plot(pc.fit, main = "Estimated CPDAG", labels=subset)
plot(pc.fit, main = "Estimated CPDAG")




#fci.fit <- fci(suffStat, indepTest, labels=subset, alpha = alpha, verbose = TRUE)
#fci.fit@amat
#plot(fci.fit)
