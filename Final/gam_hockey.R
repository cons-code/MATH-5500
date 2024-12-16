library(car)
library(dplyr)
library(glmnet)
library(ggfortify)
library(ggplot2)
library(gratia)
library(itsadug)
library(kableExtra)
library(knitr)
library(mgcv)
library(patchwork)

raw_data <- read.csv("combined.csv", header = TRUE)

raw_data$Yrs_in_NHL <- 2016 - raw_data$DftYr
raw_data$A <- raw_data$A1 + raw_data$A2
raw_data$Salary_log <- log(raw_data$Salary)

# handle missing data points
raw_data$GF[is.na(raw_data$GF)] <- median(raw_data$GF, na.rm=TRUE)
raw_data$GA[is.na(raw_data$GA)] <- median(raw_data$GA, na.rm=TRUE)
raw_data$iSF[is.na(raw_data$iSF)] <- median(raw_data$iSF, na.rm=TRUE)
raw_data$iHA[is.na(raw_data$iHA)] <- median(raw_data$iHA, na.rm=TRUE)
raw_data$Yrs_in_NHL[is.na(raw_data$Yrs_in_NHL)] <- 0

# filter out rookie players
# only retain offense
data <- raw_data[raw_data$Yrs_in_NHL > 3 & raw_data$Is_Offense == 1, ]

# Initial GAM model
model_initial <- gam(Salary_log ~ s(G) + s(A) + s(GF) + s(plus_mins) + s(TOI) 
                 + s(iSF) + s(Yrs_in_NHL) + s(Wt) + s(GP), 
                 data = data, method = "REML")
summary(model_initial)

# Output to Latex
gamtabs(model_initial, digits = 0)

# Partial Effects Plots
p1 <- draw(model_initial, residuals = T, select = smooths(model_initial)[5]) & theme_minimal() 
p2 <- draw(model_initial, residuals = T, select = smooths(model_initial)[7]) & theme_minimal()
p3 <- draw(model_initial, residuals = T, select = smooths(model_initial)[8]) & theme_minimal()
list(p1, p2, p3) |>
  wrap_plots() +
  plot_layout(nrow = 1, ncol = 3) 

p1 & theme(text = element_text(size = 20))
p2 & theme(text = element_text(size = 20))
p3 & theme(text = element_text(size = 20))

# Model diagnostics
appraise(model_initial, point_col = 'darkblue', point_alpha = 0.4,
         line_col = "black") & theme_minimal()
gam.check(model_initial)

# Model diagnostics individual charts
(qq_plot(model_initial, point_col = 'darkblue',
        line_col = "black", title = 'QQ plot of residuals', subtitle = '') & theme_minimal() 
  & theme(text = element_text(size = 20))) 
(residuals_linpred_plot(model_initial, point_col = 'darkblue',
                       line_col = "black", 
                       title='Residuals vs linear predictor', 
                       subtitle = '') & theme_minimal() 
  & theme(text = element_text(size = 20)))
(residuals_hist_plot(model_initial, 
                     title='Histogram of residuals', 
                     subtitle = '') & theme_minimal()
  & theme(text = element_text(size = 20)))
(observed_fitted_plot(model_initial, point_col = 'darkblue',
                      title='Observed vs fitted values',
                      subtitle = '') & theme_minimal()
  & theme(text = element_text(size = 20)))


# Only smooth TOI, Yrs_in_NHL
# drop Jaromir Jágr 
new_data <- data[data$Yrs_in_NHL < max(data$Yrs_in_NHL),]

model_gam_2 <- gam(Salary_log ~ G + A + GF + plus_mins + s(TOI) 
                 + iSF + s(Yrs_in_NHL) + Wt + GP, 
                 data = new_data, method = "REML")
summary(model_gam_2)
draw(model_gam_2, residuals = T)
gam.check(model_gam_2)

# Model diagnostics
appraise(model_gam_2, point_col = 'darkblue', point_alpha = 0.4,
         line_col = "black") & theme_minimal()

# Model diagnostics individual charts
(qq_plot(model_gam_2, point_col = 'darkblue',
         line_col = "black", title = 'QQ plot of residuals', subtitle = '') & theme_minimal() 
  & theme(text = element_text(size = 20))) 
(residuals_linpred_plot(model_gam_2, point_col = 'darkblue',
                        line_col = "black", 
                        title='Residuals vs linear predictor', 
                        subtitle = '') & theme_minimal() 
  & theme(text = element_text(size = 20)))
(residuals_hist_plot(model_gam_2, 
                     title='Histogram of residuals', 
                     subtitle = '') & theme_minimal()
  & theme(text = element_text(size = 20)))
(observed_fitted_plot(model_gam_2, point_col = 'darkblue',
                      title='Observed vs fitted values',
                      subtitle = '') & theme_minimal()
  & theme(text = element_text(size = 20)))


# Output to Latex
gamtabs(model_gam_2)

# Individual Partial Effects Plots
p4 <- draw(model_gam_2, residuals = T, select = smooths(model_gam_2)[1]) & theme_minimal() 
p5 <- draw(model_gam_2, residuals = T, select = smooths(model_gam_2)[2]) & theme_minimal() 

p4 & theme(text = element_text(size = 20))
p5 & theme(text = element_text(size = 20))

concurvity(model_initial)
concurvity(model_gam_2)

