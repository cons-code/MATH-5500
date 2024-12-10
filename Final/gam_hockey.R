library(car)
library(dplyr)
library(glmnet)
library(ggfortify)
library(ggplot2)
library(gratia)
library(kableExtra)
library(knitr)
library(mgcv)

raw_data <- read.csv("combined.csv", header = TRUE)

raw_data$Yrs_in_NHL <- 2016 - raw_data$DftYr
data <- raw_data[raw_data$Yrs_in_NHL > 3 & raw_data$Is_Offense == 1, ]

scale_numeric <- function(x) x %>% mutate_if(is.numeric, function(y) as.vector(scale(y)))
scaled_data <- data %>% scale_numeric()

# consider smoothing
# thinplate regression spline
model_gam <- gam(Salary_log ~ s(G) + s(A) + s(GF) + s(plus_mins) + s(TOI) 
                 + s(iSF) + s(Yrs_in_NHL) + s(Wt) + s(GP), 
                 data = scaled_data)
summary(model_gam)
par(mfrow = c(3, 3))
plot(model_gam, resid = T, shade = T)

par(mfrow = c(2, 2))
gam.check(model_gam)
concurvity(model_gam)


# Keep iSF and GP parametric
model_gam_2 <- gam(Salary_log ~ s(G) + s(A) + s(GF) + s(plus_mins) + s(TOI) 
                 + iSF + s(Yrs_in_NHL) + s(Wt) + GP, 
                 data = scaled_data)
summary(model_gam_2)
par(mfrow = c(3, 3))
plot(model_gam_2, resid = T, shade = T)

par(mfrow = c(2, 2))
gam.check(model_gam_2)

# Comparison of smoothing
# really just comparing the same smoothers
# but leveraging the visualization
comp <- compare_smooths(model_gam, model_gam_2)
draw(comp)


# compare to cubic regression spline
model_gam_cub <- gam(Salary_log ~ s(G, bs="cr") + s(A, bs="cr") + s(GF, bs="cr") 
                     + s(plus_mins, bs="cr") + s(TOI, bs="cr") 
                      + s(iSF, bs="cr") + s(Yrs_in_NHL, bs="cr") + s(Wt, bs="cr") 
                     + s(GP, bs="cr"), data = scaled_data)
summary(model_gam_cub)
par(mfrow = c(3, 3))
plot(model_gam, resid = T, shade = T)

par(mfrow = c(2, 2))
gam.check(model_gam_cub)

concurvity(model_gam_cub)

comp <- compare_smooths(model_gam, model_gam_cub)
draw(comp)
