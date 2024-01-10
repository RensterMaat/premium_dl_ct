library(tidyverse)
library(meta)

df = read.csv("/home/rens/repos/premium_dl_ct/tables/dl_vs_clinical.csv")

m = metagen(
    TE = df$diff,
    seTE=df$std,
    studlab=df$X,
    fixed=TRUE,
    random=TRUE,
)