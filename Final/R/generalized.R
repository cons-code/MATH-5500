library(car)
library(dplyr)
library(GGally)
library(ggplot2)
library(ggfortify)

df <- read.csv('../Data/combined.csv')

model <- glm(Salary ~ TOI/GP + Is_Offense + iHF + S.Slap + iPENT + CF, 
             data = df, family = Gamma(link = "log"))
autoplot(model)
coef(model)

df['High.Salary'] = 1*(df['Salary'] >= 2)
model.ge.2 <- glm(High.Salary ~ PTS + TOI/GP + Is_Offense + iHF + S.Slap + iPENT + CF,
                  data = df, family = binomial(link = "probit"))
coef(model.ge.2)
autoplot(model.ge.2)