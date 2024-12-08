library(boot)
library(car)
library(dplyr)
library(glmnet)
library(ggfortify)
library(ggplot2)
library(kableExtra)
library(knitr)

raw_data <- read.csv("combined.csv", header = TRUE)
raw_data$Yrs_in_NHL <- 2016 - raw_data$DftYr

# filter out rookie players
# only retain offense
data <- raw_data[raw_data$Yrs_in_NHL > 3 & raw_data$Is_Offense == 1, ]

model <- glm(Salary ~ G + A + GF + plus_mins + TOI + iSF + 
              Yrs_in_NHL + Wt + GP, data = data, family = Gamma(link = "log"))

summary(model)
model.diag <- glm.diag(model)
glm.diag.plots(model, model.diag)

vif(model)
mean(vif(model))

