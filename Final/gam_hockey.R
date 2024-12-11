library(car)
library(dplyr)
library(glmnet)
library(ggfortify)
library(ggplot2)
library(gratia)
library(kableExtra)
library(knitr)
library(mgcv)
library(patchwork)

raw_data <- read.csv("combined.csv", header = TRUE)

raw_data$Yrs_in_NHL <- 2016 - raw_data$DftYr
data <- raw_data[raw_data$Yrs_in_NHL > 3 & raw_data$Is_Offense == 1, ]

scale_numeric <- function(x) x %>% mutate_if(is.numeric, function(y) as.vector(scale(y)))
scaled_data <- data %>% scale_numeric()

# Initial GAM model
model_initial <- gam(Salary_log ~ s(G) + s(A) + s(GF) + s(plus_mins) + s(TOI) 
                 + s(iSF) + s(Yrs_in_NHL) + s(Wt) + s(GP), 
                 data = scaled_data, method = "REML")
summary(model_initial)

p1 <- draw(model_initial, residuals = T, select = smooths(model_initial)[4])
p2 <- draw(model_initial, residuals = T, select = smooths(model_initial)[7])
p3 <- draw(model_initial, residuals = T, select = smooths(model_initial)[8])
list(p1, p2, p3) |>
  wrap_plots() +
  plot_layout(nrow = 1, ncol = 3)

# Model diagnostics
appraise(model_gam, point_col = 'darkblue', point_alpha = 0.4,
         line_col = "black") & theme_minimal()
gam.check(model_gam)

# Only smooth TOI, Yrs_in_NHL
# drop Jaromir Jágr 
new_data <- scaled_data[scaled_data$Yrs_in_NHL < max(scaled_data$Yrs_in_NHL),]
model_gam_2 <- gam(Salary_log ~ G + A + GF + plus_mins + s(TOI) 
                 + iSF + s(Yrs_in_NHL) + Wt + GP, 
                 data = new_data, method = "REML")
summary(model_gam_2)
draw(model_gam_2, residuals = T)
gam.check(model_gam_2)

# Model diagnostics
appraise(model_gam_2, point_col = 'darkblue', point_alpha = 0.4,
         line_col = "black") & theme_minimal()



