library(dplyr)
library(ggplot2)
library(gridExtra)
library(caret)
library(Amelia)
library(mice)
library(pROC)

train <- read.csv("train.csv", sep = ",")
test <- read.csv("test.csv", sep =",")

#Adding suvived column to the test dataset

test$Survived <- 0

#Combining test and train data for cleanup
cmb <- rbind(train, test)

str(cmb)
summary(cmb)


#Creating a new variable, "Title", 
#to further simplify Name variable for exploration
cmb$Title <- gsub('(.*, )|(\\..*)', '', cmb$Name)

unique(cmb$Title)

distinct(cmb, Sex, Title)

#Based on the Sex of the passengers the 
#following errors can be rectified
cmb$Title[cmb$Title == "Mme"] <- "Mr"
cmb$Title[cmb$Title == "Ms"] <- "Miss"
cmb$Title[cmb$Title == "Mlle"] <- "Miss"


survival_title <- ggplot(cmb, aes(x=Title, fill = factor(Survived))) +
        geom_bar(position = "dodge") +
        ggtitle("Survival by Title")

plot(survival_title)
        
table(cmb$Survived, cmb$Title)


survival_age <- ggplot(cmb, aes(x=Age, fill=factor(Survived))) + 
        geom_histogram(bins = 30) + ggtitle("Survival by Age")

plot(survival_age)

survival_gender <- ggplot(cmb, aes(x=Sex, fill=factor(Survived))) + 
        geom_bar() + 
        ggtitle("Survival by Gender")

plot(survival_gender)

summary(cmb$Fare)

high_fare <- filter(cmb, Fare > 31)
mid_fare <- filter(cmb, Fare <=31 | Fare >= 7.91 )
low_fare <- filter(cmb, Fare < 7.91)

high_fare_plot <- ggplot(high_fare, aes(x=factor(Survived)))+ 
        geom_bar(aes(y= (..count..)/sum(..count..))) +
        scale_y_continuous(labels = scales::percent, limits = c(0,1)) + 
        ggtitle("Survival with High Fare") +
        ylab("Relative Survival")

mid_fare_plot <- ggplot(mid_fare, aes(x=factor(Survived)))+ 
        geom_bar(aes(y= (..count..)/sum(..count..))) +
        scale_y_continuous(labels = scales::percent, limits = c(0,1)) + 
        ggtitle("Survival with Mid Fare") + labs (y=NULL)

low_fare_plot <- ggplot(low_fare, aes(x=factor(Survived)))+ 
        geom_bar(aes(y= (..count..)/sum(..count..))) +
        scale_y_continuous(labels = scales::percent, limits = c(0,1)) +  
        ggtitle("Survival with Low Fare") + labs (y=NULL)


grid.arrange(high_fare_plot, mid_fare_plot, low_fare_plot, ncol=3)

cmb$Fare[1044] <- median(cmb$Fare, na.rm = TRUE)

aggregate(cmb$Fare, by=list(cmb$Embarked), mean)

#Since the passengers who embarked from C had a higher fare, 
#passenger 62 and 830 can be assumed to have embarked from C
cmb[c(62, 830), "Embarked"] <- "C"

missmap(cmb)

meanimpute_cmb <- cmb

mean_cmb <- aggregate(meanimpute_cmb$Age, by = list(meanimpute_cmb$Title), 
                        FUN = function(x) mean(x, na.rm = T))

meanimpute_cmb[is.na(meanimpute_cmb$Age), "Age"] <- apply(meanimpute_cmb[is.na(meanimpute_cmb$Age),], 1, 
                                        function(x) mean_cmb[mean_cmb[,1] == x["Title"], 2])

mice_mod_pmm <- mice(cmb[,!colnames(cmb) %in% 
        c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], 
        method = "pmm")

complete_pmm <- complete(mice_mod_pmm)

mice_mod_rf <- mice(cmb[,!colnames(cmb) %in% 
                                   c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], 
                     method = "rf")

complete_rf <- complete(mice_mod_rf)

plot_pmm <- ggplot(complete_pmm, aes(x=Age)) + geom_histogram(bins = 30) + 
        ggtitle("Predictive Mean Matching")

plot_rf <- ggplot(complete_rf, aes(x=Age)) + geom_histogram(bins = 30) + 
        ggtitle("Random Forest")

plot_original <- ggplot(cmb, aes(x=Age)) + geom_histogram(bins = 30) + 
        ggtitle("Orignial")

plot_meanimpute<- ggplot(meanimpute_cmb, aes(x=Age)) + geom_histogram(bins = 30) + 
        ggtitle("Mean Imputations")

grid.arrange(plot_original, plot_rf, plot_pmm, plot_meanimpute, ncol = 2, nrow = 2)


cmb$Age <- complete_rf$Age

sum(is.na(cmb$Age))

missmap(cmb)

cmb$Sex  <- factor(cmb$Sex)
cmb$Embarked  <- factor(cmb$Embarked)
cmb$Title  <- factor(cmb$Title)
cmb$Pclass  <- factor(cmb$Pclass)


##Predictive Modeling

train_new <- cmb[1:891,]
test_new <- cmb[892:1309,]

inTrain <- createDataPartition(train_new$Survived, p=0.7, list = FALSE)
trainData <- train_new[inTrain,]
testData <- train_new[-inTrain,]
set.seed(1424)

logistic_model <- glm(Survived ~ Age + Pclass + Sex + Fare + Embarked + Title,
                      data = trainData, family =binomial("logit"))
logistic_roc <- roc(trainData$Survived, predict(logistic_model, newdata = trainData, type = "response"))

plot.roc(logistic_roc, print.auc =T)

rf_model <- randomForest(factor(Survived) ~ Age + Pclass + Sex + Fare + Embarked + Title , 
                  data = trainData)
rf_predict <- predict(rf_model, testData)
confusionMatrix(table(rf_predict, testData$Survived))


final_predict <- predict(rf_model, newdata = test_new)

