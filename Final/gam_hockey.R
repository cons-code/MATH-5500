library(car)
library(mgcv)
library(mgcv.helper)
library(dplyr)
library(glmnet)
library(ggfortify)
library(ggplot2)
library(kableExtra)
library(knitr)
library(splines)
library(splines2)

raw_data <- read.csv("combined.csv", header = TRUE)

raw_data$Yrs_in_NHL <- 2016 - raw_data$DftYr
data <- raw_data[raw_data$Yrs_in_NHL > 3 & raw_data$Is_Offense == 1, ]

# consider smoothing
model_gam <- gam(Salary_log ~ s(G) + s(A) + s(GF) + s(plus_mins) + s(TOI) 
                 + s(iSF) + s(Yrs_in_NHL) + s(Wt) + s(GP), 
                 data = data)
summary(model_gam)
plot(model_gam, resid = T, shade = T)
concurvity(model_gam)

