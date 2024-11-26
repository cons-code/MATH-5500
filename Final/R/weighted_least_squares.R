library(car)
library(GGally)
library(ggplot2)

df <- read.csv('../Data/combined.csv')

# +/- is rendered as X...
# E+/- is rendered as E...
model_R1_log <- lm(log(Salary) ~ GP + PTS + X... + Grit + PIM, data = df)
summary(model_R1_log)
autoplot(model_R1_log)

e_i_R1    <- resid(model_R1_log)
y_hat_R1  <- predict(model_R1_log)
k_hat     <- summary(model_R1_log)$sigma^2

plot(y_hat_R1, abs(rstudent(model_R1_log)))
plot(df$GP, abs(rstudent(model_R1_log)))
plot(df$PTS, abs(rstudent(model_R1_log)))
plot(df$X..., abs(rstudent(model_R1_log)))
plot(df$Grit, abs(rstudent(model_R1_log)))
plot(df$PIM, abs(rstudent(model_R1_log)))

# Try with Games Played (GP)
##### standard deviation function #####
sd_fun_data   <- data.frame(y = abs(e_i_R1), x = df$GP)
model_R1_sdf  <- lm(y ~ x, data = sd_fun_data)
sigma_hat     <- predict(model_R1_sdf)
w_i_sd        <- 1/(sigma_hat^2)

model_R1_sd_wls <- lm(log(Salary) ~ GP + PTS + X... + Grit  + PIM, 
                      data = df, weights = w_i_sd)
round(coef(model_R1_log), 5)
round(coef(model_R1_sd_wls), 5)
autoplot(model_R1_sd_wls)
mean(vif(model_R1_sd_wls))
summary(model_R1_sd_wls)