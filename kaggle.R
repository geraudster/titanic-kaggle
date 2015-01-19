# Load packages
install.packages(c('psych',
                   'car',
                   'lsr',
                   'ggplot2',
                   'reshape',
                   'aod',
                   'QuantPsyc'))

library(psych)
library(car)
library(lsr)
library(ggplot2)
library(reshape)
library(aod)
library(QuantPsyc)

train <- read.table("train.csv", header=T, sep=",")
train$Survived <- factor(train$Survived,levels=c(0,1), labels=c("Dead", "Survived"))
train$Pclass <- factor(train$Pclass)
train.sex <- C(train$Sex, treatment)
train.class <- C(train$Pclass)
train$Age[is.na(train$Age)] <- mean(train$Age, na.rm=T)

describe(train)
describeBy(train, train$Survived)
describeBy(train, train.sex)


lrfit <- glm(Survived ~ train.class + train.sex + Fare + SibSp + Parch + Age, family = binomial, data=train)
summary(lrfit)
confint(lrfit)

with(lrfit, null.deviance - deviance) #difference in deviance for the two models
with(lrfit, df.null - df.residual) #df for the difference between the two models
with(lrfit, pchisq(null.deviance-deviance, df.null-df.residual, lower.tail = FALSE)) #p-value

coef(lrfit)
wald.test(b = coef(lrfit), Sigma = vcov(lrfit), Terms = 2) #class2
wald.test(b = coef(lrfit), Sigma = vcov(lrfit), Terms = 3) #class3
wald.test(b = coef(lrfit), Sigma = vcov(lrfit), Terms = 4) #sexmale
wald.test(b = coef(lrfit), Sigma = vcov(lrfit), Terms = 5) #fare
wald.test(b = coef(lrfit), Sigma = vcov(lrfit), Terms = 6) #sibsp
wald.test(b = coef(lrfit), Sigma = vcov(lrfit), Terms = 7) #parch
wald.test(b = coef(lrfit), Sigma = vcov(lrfit), Terms = 8) #age

# Odds ratios
exp(coef(lrfit))

ClassLog(lrfit, train$Survived)

# predict(lrfit)

test <- read.table("test.csv", header=T, sep=",")
test$Survived <- factor(test$Survived,levels=c(0,1), labels=c("Dead", "Survived"))
test$Pclass <- factor(test$Pclass)
test.sex <- C(test$Sex, treatment)
test.class <- C(test$Pclass)
test$Age[is.na(test$Age)] <- mean(test$Age, na.rm=T)
test$Fare[is.na(test$Fare)] <- mean(test$Fare, na.rm=T)
newdata <- data.frame(test,train.sex=test.sex,train.class=test.class)
predicted <- melt(predict(lrfit, newdata, type = "response" ))

prediction <- data.frame(test$PassengerId, round(predicted))
prediction <- rename(prediction, c(test.PassengerId="PassengerId", value="Survived"))
write.table(prediction, file="prediction_R_logit.csv", sep=',', row.names=F, quote=F)
ggplot(predicted, aes(x=value)) + geom_bar()

fit <- melt(predict(lrfit, type="response"))
ggplot(fit, aes(x=value)) + geom_bar()

describe(prediction)

# second

lrfit2 <- glm(Survived ~ train.class + train.sex + SibSp + Age, family = binomial, data=train)
summary(lrfit2)
confint(lrfit2)

with(lrfit2, null.deviance - deviance) #difference in deviance for the two models
with(lrfit2, df.null - df.residual) #df for the difference between the two models
with(lrfit2, pchisq(null.deviance-deviance, df.null-df.residual, lower.tail = FALSE)) #p-value
newdata <- data.frame(test,train.sex=test.sex,train.class=test.class)
predicted <- melt(predict(lrfit2, newdata, type = "response" ))

prediction <- data.frame(test$PassengerId, round(predicted))
prediction <- rename(prediction, c(test.PassengerId="PassengerId", value="Survived"))
write.table(prediction, file="prediction_R_logit_2.csv", sep=',', row.names=F, quote=F)
ggplot(predicted, aes(x=value)) + geom_bar()