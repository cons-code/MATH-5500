library(car)
library(dplyr)
library(GGally)
library(ggplot2)
library(ggfortify)
library(glmnet)

df <- read.csv('../Data/combined.csv')

# LASSO
exclude <- c('Ovrl', 'DftRd', 'DftYr')
clean <- df[, !names(df) %in% exclude]
clean <- na.omit(clean)
clean <- clean %>% dplyr::select(where(is.numeric))

X <- clean[, !names(clean) %in% c('Salary')]
Y <- log(clean$Salary)

# LASSO
model_lasso_cv  <- cv.glmnet(x = X, y = Y, family = "gaussian", alpha = 1,
                             type.measure = "mse", nfolds = 5,
                             grouped = FALSE)
