library(car)
library(dplyr)
library(glmnet)
library(ggfortify)
library(ggplot2)
library(gt)
library(gtsummary)
library(kableExtra)
library(knitr)
library(xtable)

raw_data <- read.csv("combined.csv", header = TRUE)

raw_data$Yrs_in_NHL <- 2016 - raw_data$DftYr
raw_data$A <- raw_data$A1 + raw_data$A2
raw_data$Salary_log <- log(raw_data$Salary)
raw_data$Is_Big_6 <- 1*(raw_data$Nat %in% c('CAN', 'USA', 'CZE', 'FIN', 'RUS', 'SWE'))

# handle missing data points
raw_data$GF[is.na(raw_data$GF)] <- median(raw_data$GF, na.rm=TRUE)
raw_data$GA[is.na(raw_data$GA)] <- median(raw_data$GA, na.rm=TRUE)
raw_data$iSF[is.na(raw_data$iSF)] <- median(raw_data$iSF, na.rm=TRUE)
raw_data$iHA[is.na(raw_data$iHA)] <- median(raw_data$iHA, na.rm=TRUE)
raw_data$Yrs_in_NHL[is.na(raw_data$Yrs_in_NHL)] <- 0

# filter out rookie players
# only retain offense
data <- raw_data[raw_data$Yrs_in_NHL > 3 & raw_data$Is_Offense == 1, ]

# supporting visualizations
# Years in NHL
hist(raw_data$Yrs_in_NHL,
     main = "Years in NHL",
     xlab = "Years in NHL")

# Players Drafted by Year
barplot(table(raw_data$DftYr),
        main = "Players Drafted by Year",
        xlab = "Year",
        ylab = "Count")

# Salary by Position Type
raw_data$Is_Offense <- as.factor(raw_data$Is_Offense)
raw_data |>
  filter(Yrs_in_NHL > 3) |>
  ggplot(aes(x=Salary)) +
    geom_histogram(alpha=0.5, position='identity') +
    facet_grid(Is_Offense ~ .)

# Logged Salary by Position Type
raw_data |>
  filter(Yrs_in_NHL > 3) |>
  ggplot(aes(x=Salary_log)) +
    geom_histogram(alpha=0.5, position='identity') +
    facet_grid(Is_Offense ~ .)  


# summary stats
cols <- c('A', 'CA', 'CF', 'FA', 'FF', 'FO_pct', 'G', 'GA', 'GF', 'GP','Grit','Ht',
          'iBLK', 'iHA', 'iHF', 'iSF', 'PIM', 'plus_mins',
          'Salary', 'TOI', 'Wt', 'xGA', 'xGF', 'Yrs_in_NHL')
min_ <- (apply(data[cols], 2, min))
max_ <- (apply(data[cols], 2, max))
mean_ <- (apply(data[cols], 2, mean))
sum.stats <- data.frame(cbind(mean=mean_, min=min_, max=max_))

sum.stats %>%
#  round(2) %>%
  kbl(caption = paste("Summary Statistics (n =", nrow(data), ")", sep=''), 
      label = "summary_stats",
      format = "latex",
      booktabs = T,
      linesep = "",
      digits = 1)

# baseline model
base <- lm(Salary_log ~ FO_pct + G + A + GF + GA + plus_mins + TOI + iSF 
           + Yrs_in_NHL + Ht + Wt + Grit + iHF + iHA + iBLK + PIM + GP, data = data)
summary(base)
mean(vif(base)) # 469.5
autoplot(base)

# baseline2 model
base2 <- lm(Salary_log ~ G + A + GF + plus_mins + TOI + iSF + 
              Yrs_in_NHL + Wt + GP, data = data)
summary(base2)
mean(vif(base2)) # 13.47
autoplot(base2)

# Adding Is_Big_6 does nothing
base3 <- lm(Salary_log ~ G + A + GF + plus_mins + TOI + iSF + 
              Yrs_in_NHL + Wt + GP + Is_Big_6, data = data)
summary(base3)

# Overall Draft position
base4 <- lm(Salary_log ~ G + A + GF + plus_mins + TOI + iSF + 
              Yrs_in_NHL + Wt + GP + Ovrl, data = data)
summary(base4)
mean(vif(base4)) # 12.35
autoplot(base4)

# Draft Year not populated for 125 players
sum(is.na(raw_data$DftYr))

# Model defense?
def <- raw_data[raw_data$Yrs_in_NHL > 3 & raw_data$Is_Offense == 0, ]
base_def <- lm(Salary_log ~ FO_pct + G + A + GF + GA + plus_mins + TOI + iSF 
               + Yrs_in_NHL + Ht + Wt + Grit + iHF + iHA + iBLK + PIM + GP, data = def)
summary(base_def)
mean(vif(base_def))
autoplot(base_def)

base_def2 <- lm(Salary_log ~ GA + plus_mins + TOI + iBLK
                + Yrs_in_NHL + GP + Grit + iHF + Ovrl, data = def)
summary(base_def2)
mean(vif(base_def2)) # 18.6
autoplot(base_def2)

# Ovechkin Stats
# Retrieved manually but possibly code here

# Scaled models
scale_numeric <- function(x) x %>% mutate_if(is.numeric, function(y) as.vector(scale(y)))
scaled_data <- data %>% scale_numeric()

# bedrock model, non-logged
bedrock_no_log <- lm(Salary ~ FO_pct + G + A + GF + GA 
                    + plus_mins + TOI + iSF + Yrs_in_NHL 
                    + Ht + Wt + Grit + iHF + iHA 
                    + iBLK + PIM + GP, data = scaled_data)
summary(bedrock_no_log)
autoplot(bedrock_no_log)


# bedrock model, minus Grit
bedrock <- lm(Salary_log ~ FO_pct + G + A + GF + GA 
              + plus_mins + TOI + iSF + Yrs_in_NHL 
              + Ht + Wt + iHF + iHA 
              + iBLK + PIM + GP, data = scaled_data)
summary(bedrock)
vif(bedrock)
mean(vif(bedrock))
s <- summary(bedrock)$coefficients[,c(1,3,4)]

print(xtable(cbind(s, VIF = vif(bedrock))),
      floating=FALSE,latex.environments=NULL,booktabs=TRUE)

