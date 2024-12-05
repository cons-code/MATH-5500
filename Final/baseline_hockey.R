library(car)
library(dplyr)
library(glmnet)
library(ggfortify)
library(ggplot2)
library(kableExtra)
library(knitr)

data = read.csv("combined.csv", header = TRUE)

data$Yrs_in_NHL <- 2016 - data$DftYr
data$A <- data$A1 + data$A2
data$Salary_log <- log(data$Salary)

# handle missing data points
data$GF[is.na(data$GF)] <- median(data$GF, na.rm=TRUE)
data$GA[is.na(data$GA)] <- median(data$GA, na.rm=TRUE)
data$iSF[is.na(data$iSF)] <- median(data$iSF, na.rm=TRUE)
data$iHA[is.na(data$iHA)] <- median(data$iHA, na.rm=TRUE)
data$Yrs_in_NHL[is.na(data$Yrs_in_NHL)] <- 0

# filter out rookie players
# only retain offense
data <- data[data$Yrs_in_NHL > 3,]
data <- data[data$Is_Offense == 1,]

# summary stats
cols <- c('A', 'FO_pct', 'G', 'GA', 'GF', 'GP','Grit','Ht',
          'iBLK', 'iHA', 'iHF', 'iSF', 'PIM', 'plus_mins',
          'Salary', 'TOI', 'Wt', 'Yrs_in_NHL')
min_ <- (apply(data[cols], 2, min))
max_ <- (apply(data[cols], 2, max))
mean_ <- (apply(data[cols], 2, mean))
sum.stats <- data.frame(cbind(mean=mean_, min=min_, max=max_))

sum.stats %>%
  round(2) %>%
  kbl(caption = paste("Summary Statistics (n =", nrow(data), ")", sep=''), 
      label = "summary_stats",
      format = "latex",
      booktabs = T,
      linesep = "")

# baseline model
base <- lm(Salary_log ~ FO_pct + G + A + GF + GA + plus_mins + TOI + iSF 
           + Yrs_in_NHL + Ht + Wt + Grit + iHF + iHA + iBLK + PIM + GP, data = data)
summary(base)
mean(vif(base)) # 469.5
autoplot(base)

# baseline2 model
base2 <- lm(Salary_log ~ G + A + GF + GA + plus_mins + TOI + iSF + 
              Yrs_in_NHL + GP, data = data)
summary(base2)
mean(vif(base2)) # 17.77
autoplot(base2)





