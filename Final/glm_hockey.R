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

# Gamma GLM
model <- glm(Salary ~ G + A + GF + plus_mins + TOI + iSF + 
              Yrs_in_NHL + Wt + GP, data = data, family = Gamma(link = "log"))

summary(model)
model.diag <- glm.diag(model)
glm.diag.plots(model, model.diag)

vif(model)
mean(vif(model))

# Inverse Gaussian
# no-go
gaussian_inv <- glm(Salary ~ G + A + GF + plus_mins + TOI + iSF + 
                      Yrs_in_NHL + Wt + GP, data = data, 
                    family = gaussian(link = "inverse"))
summary(gaussian_inv)
gaussian_inv.diag <- glm.diag(gaussian_inv)
glm.diag.plots(gaussian_inv, gaussian_inv.diag)
