### FINAL PROJECT: GLS ###

## LIBRARY INSTALLATION ##

library(car)
library(dplyr)
library(nlme)
library(glmnet)
library(xtable)


## DATA PROCESSING ##

# Read CV containing combined NHL dataframe.
nhl_df <- read.csv("./Georgetown/MATH 5500/final_project/combined.csv", 
                   header=TRUE)

# Perform data cleaning.
nhl_df$Yrs_in_NHL <- 2016 - nhl_df$DftYr
nhl_df$A          <- nhl_df$A1 + nhl_df$A2

nhl_df$GF[is.na(nhl_df$GF)]   <- median(nhl_df$GF, na.rm=TRUE)
nhl_df$GA[is.na(nhl_df$GA)]   <- median(nhl_df$GA, na.rm=TRUE)

nhl_df$iSF[is.na(nhl_df$iSF)] <- median(nhl_df$iSF, na.rm=TRUE)
nhl_df$iHA[is.na(nhl_df$iHA)] <- median(nhl_df$iHA, na.rm=TRUE)

nhl_df$Yrs_in_NHL[is.na(nhl_df$Yrs_in_NHL)] <- 0

nhl_df <- nhl_df[nhl_df$Yrs_in_NHL > 3,]
nhl_df <- nhl_df[nhl_df$Is_Offense == 1,]

scale_numeric <- function(x) x %>% mutate_if(is.numeric, function(y) as.vector(scale(y)))
nhl_df_scaled <- nhl_df %>% scale_numeric()

# Extract baseline parameters from scaled NHL dataframe.
nhl_df_base <- data.frame(Salary_log=nhl_df_scaled$Salary_log,
                          FO_pct=nhl_df_scaled$FO_pct,
                          G=nhl_df_scaled$G,
                          A=nhl_df_scaled$A,
                          GF=nhl_df_scaled$GF,
                          GA=nhl_df_scaled$GA,
                          plus_mins=nhl_df_scaled$plus_mins,
                          TOI=nhl_df_scaled$TOI,
                          iSF=nhl_df_scaled$iSF,
                          Yrs_in_NHL=nhl_df_scaled$Yrs_in_NHL,
                          Ht=nhl_df_scaled$Ht,
                          Wt=nhl_df_scaled$Wt,
                          Grit=nhl_df_scaled$Grit,
                          iHF=nhl_df_scaled$iHF,
                          iHA=nhl_df_scaled$iHA,
                          iBLK=nhl_df_scaled$iBLK,
                          PIM=nhl_df_scaled$PIM,
                          GP=nhl_df_scaled$GP)

# Extract advanced parameters from scaled NHL dataframe.
nhl_df_adv <- data.frame(nhl_df_base)

nhl_df_adv$PDO <- nhl_df_scaled$PDO

nhl_df_adv$CA  <- nhl_df_scaled$CA
nhl_df_adv$CF  <- nhl_df_scaled$CF

nhl_df_adv$FA  <- nhl_df_scaled$FA
nhl_df_adv$FF  <- nhl_df_scaled$FF

nhl_df_adv$xGA <- nhl_df_scaled$xGA
nhl_df_adv$xGF <- nhl_df_scaled$xGF

# Generate pairs plot of baseline NHL parameters.
pairs(nhl_df_base)


## OLS ANALYSIS ##

# Generate simple linear model with baseline metrics (post-penalization).
nhl_mdl_ols_base <- lm(Salary_log ~ A + GF + plus_mins + TOI + iSF 
                       + Yrs_in_NHL + Wt + GP,
                       data=nhl_df_base)

# Generate simple linear model with advanced metrics.
nhl_mdl_ols_adv <- lm(Salary_log ~ A + GF + plus_mins + TOI + iSF 
                      + Yrs_in_NHL + Wt + GP
                      + PDO + CA + CF + FA + FF + xGA + xGF
                      + (CA*CF) + (FA*FF) + (CA*FA) + (CF*FF) 
                      + (xGA*xGF) + (PDO*xGA) + (PDO*xGF),
                      data=nhl_df_adv)

# Perform post-analysis.
summary(nhl_mdl_ols_base)
confint(nhl_mdl_ols_base)

vif(nhl_mdl_ols_base)
mean(vif(nhl_mdl_ols_base))

plot(nhl_mdl_ols_base)

summary(nhl_mdl_ols_adv)
confint(nhl_mdl_ols_adv)

vif(nhl_mdl_ols_adv)
mean(vif(nhl_mdl_ols_adv))

plot(nhl_mdl_ols_adv)


## PENALIZED REGRESSION ANALYSIS ##

# Perform LASSO penalized regression with advanced metrics.
X <- model.matrix(Salary_log ~ A + GF + plus_mins + TOI + iSF 
                  + Yrs_in_NHL + Wt + GP
                  + PDO + CA + CF + FA + FF + xGA + xGF
                  + (CA*CF) + (FA*FF) + (CA*FA) + (CF*FF) 
                  + (xGA*xGF) + (PDO*xGA) + (PDO*xGF),
                  data=nhl_df_adv)
Y <- nhl_df_adv$Salary_log

nhl_mdl_lasso_adv_cv <- cv.glmnet(x=X, y=Y, family="gaussian", alpha=1, 
                                type.measure="mse", nfolds=nrow(nhl_df_adv),
                                grouped=FALSE)

# Perform post-analysis.
print(nhl_mdl_lasso_adv_cv)
coef(nhl_mdl_lasso_adv_cv)

plot(nhl_mdl_lasso_adv_cv)


## GLS ANALYSIS ##

# Perform residual analysis with predictors of interest.
e  <- resid(nhl_mdl_ols_adv)
s  <- summary(nhl_mdl_ols_adv)$sigma

es <- e/s

yhat = predict(nhl_mdl_ols_adv)
plot(yhat, es, 
     type='p', xlab='yhat', ylab='es')

plot(nhl_df_adv$Yrs_in_NHL, es, 
     type='p', xlab='Years in NHL', ylab='Standardized Residuals')
plot(nhl_df_adv$TOI, es, 
     type='p', xlab='Time on Ice', ylab='Standardized Residuals')
plot(nhl_df_adv$GP, es, 
     type='p', xlab='Games Played', ylab='Standardized Residuals')

plot(nhl_df_base$FO_pct, es, 
     type='p', xlab='Percent Faceoffs Taken', ylab='Standardized Residuals')
plot(nhl_df_base$Yrs_in_NHL, es, 
     type='p', xlab='Years in NHL', ylab='Standardized Residuals')
plot(nhl_df_base$G, es, 
     type='p', xlab='Goals', ylab='Standardized Residuals')
plot(nhl_df_base$A, es, 
     type='p', xlab='Assists', ylab='Standardized Residuals')
plot(nhl_df_base$GF, es, 
     type='p', xlab='On-Ice Goals For', ylab='Standardized Residuals')
plot(nhl_df_base$GA, es, 
     type='p', xlab='On-Ice Goals Against', ylab='Standardized Residuals')
plot(nhl_df_base$plus_mins, es, 
     type='p', xlab='Plus/Minus', ylab='Standardized Residuals')
plot(nhl_df_base$TOI, es, 
     type='p', xlab='Time on Ice', ylab='Standardized Residuals')
plot(nhl_df_base$iSF, es, 
     type='p', xlab='Individual Shots on Goal', ylab='Standardized Residuals')
plot(nhl_df_base$Ht, es, 
     type='p', xlab='Height', ylab='Standardized Residuals')
plot(nhl_df_base$Wt, es, 
     type='p', xlab='Weight', ylab='Standardized Residuals')
plot(nhl_df_base$Grit, es, 
     type='p', xlab='Grit', ylab='Standardized Residuals')
plot(nhl_df_base$iHF, es, 
     type='p', xlab='Individual Hits Thrown', ylab='Standardized Residuals')
plot(nhl_df_base$iHA, es, 
     type='p', xlab='Individual Hits Taken', ylab='Standardized Residuals')
plot(nhl_df_base$iBLK, es, 
     type='p', xlab='Individual Shots Blocked', ylab='Standardized Residuals')
plot(nhl_df_base$PIM, es, 
     type='p', xlab='Penalty Minutes', ylab='Standardized Residuals')
plot(nhl_df_base$GP, es, 
     type='p', xlab='Games Played', ylab='Standardized Residuals')

plot(nhl_df_adv$CA, es, 
     type='p', xlab='Shot Attempts Allowed (Corsi)', ylab='Standardized Residuals')
plot(nhl_df_adv$CF, es, 
     type='p', xlab='Shot Attempts Taken (Corsi)', ylab='Standardized Residuals')
plot(nhl_df_adv$FA, es, 
     type='p', xlab='Unblocked Shot Attempts Allowed (Fenwick)', ylab='Standardized Residuals')
plot(nhl_df_adv$FF, es, 
     type='p', xlab='Unblocked Shot Attempts Taken (Fenwick)', ylab='Standardized Residuals')
plot(nhl_df_adv$CA, es, 
     type='p', xlab='Expected Goals Allowed (xG)', ylab='Standardized Residuals')
plot(nhl_df_adv$CF, es, 
     type='p', xlab='Expected Goals Taken (xG)', ylab='Standardized Residuals')

# Generate GLS model.
nhl_mdl_gls_base <- gls(Salary_log ~ A + GF + plus_mins + TOI + iSF 
                        + Yrs_in_NHL + Wt + GP,
                        data=nhl_df_adv,
                        weights=varIdent(form=~1|GP),
                        correlation=corAR1(form=~1))

nhl_mdl_gls_adv  <- gls(Salary_log ~ A + GF + plus_mins + TOI + iSF 
                        + Yrs_in_NHL + Wt + GP
                        + PDO + CA + CF + FA + FF
                        + (CA*FA) + (CF*FF),
                        data=nhl_df_adv,
                        weights=varIdent(form=~1|GP),
                        correlation=corAR1(form=~1))

# Perform post-analysis.
summary(nhl_mdl_gls_base)
confint(nhl_mdl_gls_base)

plot(nhl_mdl_gls_base)
plot(nhl_mdl_ols_base)

summary(nhl_mdl_gls_adv)
confint(nhl_mdl_gls_adv)

plot(nhl_mdl_gls_adv)
plot(nhl_mdl_ols_adv)

