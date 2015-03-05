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

prepareData <- function (file, hasLabel = TRUE, ageForNA, fareForNA) {
  dataset <- read.table(file, header=T, sep=",", stringsAsFactors = FALSE)
  
  if(hasLabel) {
    dataset$Survived <- factor(dataset$Survived)
    dataset$SurvivedLabel <- factor(dataset$Survived,levels=c(0,1), labels=c("Dead", "Survived"))
  }
  dataset$Pclass <- factor(dataset$Pclass)
  dataset$sex <- C(factor(dataset$Sex), treatment)
  dataset$class <- C(dataset$Pclass)
  
  if(hasLabel) {
    ageForNA <- mean(dataset$Age, na.rm=T)
    fareForNA <- mean(dataset$Fare, na.rm=T)
  }
  ## impute missing age values
  
  dataset$Age[is.na(dataset$Age)] <- ageForNA
  dataset$Fare[is.na(dataset$Fare)] <- fareForNA
  
  list(dataset, ageForNA, fareForNA)
}

trainList <- prepareData('train.csv')
train <- trainList[[1]]


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



lrfit3 <- glm(Survived ~ train.class + train.sex + Fare + SibSp + Parch + Age + train.sex * SibSp, family = binomial, data=train)
summary(lrfit3)
confint(lrfit3)
anova(lrfit, lrfit2, test = "Chisq")
anova(lrfit, lrfit3, test = "Chisq")
lrfit4 <- glm(Survived ~ train.class + train.sex + Fare + SibSp + Parch + Age + train.sex * SibSp + train.class * Fare, family = binomial, data=train)
anova(lrfit, lrfit4, test = "Chisq")

with(lrfit3, null.deviance - deviance) #difference in deviance for the two models
with(lrfit3, df.null - df.residual) #df for the difference between the two models
with(lrfit3, pchisq(null.deviance-deviance, df.null-df.residual, lower.tail = FALSE)) #p-value

coef(lrfit3)
wald.test(b = coef(lrfit3), Sigma = vcov(lrfit3), Terms = 2) #class2
wald.test(b = coef(lrfit3), Sigma = vcov(lrfit3), Terms = 3) #class3
wald.test(b = coef(lrfit3), Sigma = vcov(lrfit3), Terms = 4) #sexmale
wald.test(b = coef(lrfit3), Sigma = vcov(lrfit3), Terms = 5) #fare
wald.test(b = coef(lrfit3), Sigma = vcov(lrfit3), Terms = 6) #sibsp
wald.test(b = coef(lrfit3), Sigma = vcov(lrfit3), Terms = 7) #parch
wald.test(b = coef(lrfit3), Sigma = vcov(lrfit3), Terms = 8) #age

# Odds ratios
exp(coef(lrfit3))

ClassLog(lrfit3, train$Survived)

newdata <- data.frame(test,train.sex=test.sex,train.class=test.class)
predicted <- melt(predict(lrfit3, newdata, type = "response" ))

prediction <- data.frame(test$PassengerId, round(predicted))
prediction <- rename(prediction, c(test.PassengerId="PassengerId", value="Survived"))
write.table(prediction, file="prediction_R_logit_3.csv", sep=',', row.names=F, quote=F)
ggplot(predicted, aes(x=value)) + geom_bar()

## With Caret

install.packages(c('caret', 'randomForest', 'gbm', 'doMC', 'e1071'))
library(caret)
require('doMC')
registerDoMC(cores = 2)

testModel <- function(train, test, method) {
  model <- list()
  model$fit <- train(Survived ~ .,
                     data = train[,-c(1,3:5,9,11,13)],
                     method=method)
  model$predictions <- predict(model$fit, newdata = test[,-c(1,4,8,10)])
  model$predictionsdf <- data.frame(PassengerId = test$PassengerId, Survived = model$predictions)
  model
}

newdata <- prepareData('test.csv', hasLabel = FALSE, trainList[[2]], trainList[[3]])[[1]]

# traintt <- dcast( train, PassengerId ~ Ticket, length)

model.rf <- testModel(train, newdata, method="rf")
# model.rf$predictions <- predict(model.rf$fit, newdata = newdata[,-c(1,4,8,10)])
model.gbm <- testModel(train, newdata, method="gbm")
# model.gbm$predictions <- predict(model.gbm$fit, newdata = newdata[,-c(1,4)], type = 'prob')

write.table(model.rf$predictionsdf, file="prediction_randomforest.csv", sep=',', row.names=F, quote=F)
write.table(model.gbm$predictionsdf, file="prediction_gbm.csv", sep=',', row.names=F, quote=F)

model.rf.predict.train <- predict(model.rf, newdata = train)
model.gbm.predict.train <- predict(model.gbm, newdata = train)

modelsDf <- data.frame(model.rf.predict.train, model.gbm.predict.train, Survived = train$Survived)

modelCombo <- train(Survived ~ ., data = modelsDf, method = 'rf')
newdata <- data.frame(model.rf.predict.train = model.rf.predict, model.gbm.predict.train = model.gbm.predict)
modelCombo.predict <- predict(modelCombo, newdata = newdata)
modelCombo.predict.df <- data.frame(PassengerId = test$PassengerId, Survived = modelCombo.predict)
write.table(modelCombo.predict.df, file="prediction_combo.csv", sep=',', row.names=F, quote=F)

## test with text mining
install.packages('tm')
library(tm)
myCorpus <- Corpus(VectorSource(train$Name))
dtm <- DocumentTermMatrix(myCorpus)
train.merge <- cbind(train, inspect(dtm))

testCorpus <- Corpus(VectorSource(newdata$Name))
test.dtm <- DocumentTermMatrix(testCorpus, list(dictionary = colnames(dtm)))
test.merge <- cbind(newdata, inspect(test.dtm))

model.rf.tm <- testModel(train.merge, test.merge, method="rf")
model.gbm.tm <- testModel(train.merge, test.merge, method="gbm")

write.table(model.gbm.tm$predictionsdf, file="prediction_gbm_tm.csv", sep=',', row.names=F, quote=F)

model.evtree.tm <- testModel(train.merge, test.merge, method="evtree")
write.table(model.evtree.tm$predictionsdf, file="prediction_evtree_tm.csv", sep=',', row.names=F, quote=F)

model.evtree <- testModel(train, newdata, method="evtree")
write.table(model.evtree$predictionsdf, file="prediction_evtree.csv", sep=',', row.names=F, quote=F)
