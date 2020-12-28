##################################################
##################################################
#Intro: This code is created to model New York State Inmate data in order to 
#gain a further understanding of why inmates might be placed in a maximum 
#security facility.
#Author: Elisabeth Jones
#Last update: 12-8-20
##################################################
##################################################

#read in packages
library('psych')
library('dplyr')
library('caret')
library('doParallel')
library('pROC')
library('VIM')
library('mice')
library("rpart.plot")
library("randomForest")
library("class")

#read in data
setwd('/Users/elisabethjones/GoogleDrive/MIS620/Project/data')
imt.data <- read.csv('2019inmates.csv', stringsAsFactors = TRUE)

#understanding the data
variable.names(imt.data)
str(imt.data)

#######################################
#Parallel processing
#######################################
getDoParWorkers() 
#I only have 1 - I will not be using parallel processing for these models

#######################################
#Basic cleaning & descriptives, N=46037
#######################################
#recode dependent variable - Max Security Assigned Max/Not
imt.data$MaxSF <-  imt.data$Facility.Security.Level
sec.key <- c("MAXIMUM SECURITY" = "max", "MEDIUM  SECURITY" = "not", "MEDIUM SECURITY" = "not",
             "MINIMUM SECURITY" = "not", "SHOCK INCARCERATION" = "not")
imt.data$MaxSF <- recode(imt.data$MaxSF, !!!sec.key) 

#dept var: MaxSF
#indt var: Race/Ethnicity & Gender & latest admission type
d.var <- imt.data$MaxSF
i.var.r <- imt.data$Race.Ethnicity
i.var.g <- imt.data$Gender
i.var.a <- imt.data$Latest.Admission.Type

#descriptives of indp and dept vars
summary(d.var)
summary(i.var.r)
summary(i.var.g)
summray(i.var.a)
table(d.var) #max: 20565 (45%), not: 25472 (55%), slight imbalance.  
table(i.var.r)
table(i.var.g)
table(i.var.a)

#######################################
#Plotting descriptives to better understand data
#######################################
#bar plot
barrace <- dummy.code(imt.data$Race.Ethnicity)
bargend <- dummy.code(imt.data$Gender)
baradmis <- dummy.code(imt.data$Latest.Admission.Type)
barplot(barrace)
barplot(bargend)
barplot(baradmis)

#cross tabs
table(imt.data$MaxSF, imt.data$Race.Ethnicity) #security level x race
table(imt.data$MaxSF, imt.data$Gender)# security level x gender
table(imt.data$MaxSF, imt.data$Latest.Admission.Type)# security level x admistype

#######################################
#Converting data into dummy code and preprocessing
#######################################
#subset data to get only the relevant variables
imt.data.ss <- subset(imt.data, select = c("Gender", "MaxSF", "Race.Ethnicity",
                                           "Latest.Admission.Type")) 
#separating out dept and indp vars
y <- imt.data.ss$MaxSF #dept var
x <- imt.data.ss[,-2] #indpt vars

#dummy model
imt.dummy.model <- dummyVars("~ .", data=x, fullRank=FALSE)
#apply model to data with predict function
imt.data.dum <- data.frame(predict(imt.dummy.model, x))
#compare to original data set
str(imt.data.dum)
str(imt.data.ss)

#check for missing data
md.pattern(imt.data.dum) #no missing data

#preprocessing ignored(0), no need for bagImpute, scale, or center to replace missing data. 
imt.prepmodel <- preProcess(imt.data.dum, 
                            method=c("center", "scale","zv", "corr")) 
imt.prepmodel
imt.prep <- predict(imt.prepmodel, imt.data.dum)
str(imt.prep)

#######################################
#setting up data to be modeled
####################################### 
set.seed(348)
#lets grab 85% of data for training and 15% for test sets
inTrain<-createDataPartition(y=y, p=.85, list=FALSE) 

#lets split out using index of training and test sets created above, uses row index
y.train <- factor(y[inTrain], levels= c("max","not"), labels=c("max", "not"))
x.train <- imt.prep[inTrain,]
y.test<- factor(y[-inTrain], levels= c("max","not"), labels=c("max", "not"))
x.test <- imt.prep[-inTrain,]

#check composition
table(y.train) # max 17481 (45%) not 21652 (55%)
table(y.test) # max 3084 (45%) not 3820 (55%)


#some parameters to control the sampling during parameter tuning and testing
#5 fold crossvalidation, using 5-folds instead of 10 to reduce computation time in class demo, use 10 and with more computation to spare use
#repeated cv
ctrl <- trainControl(method="cv", number=10,
                     classProbs=TRUE,
                     #function used to measure performance
                     summaryFunction = twoClassSummary, #classification
                     allowParallel =  FALSE) #not using parallel processing for this

#######################################
#Modeling, positive class = max
####################################### 

####rpart method (decision tree)####
set.seed(348)
modelLookup("rpart")
m.rpart <- train(y=y.train, x=x.train,
                 trControl = ctrl,
                 metric = "ROC", #using AUC to find best performing parameters
                 tuneLength = 15,
                 method = "rpart")
m.rpart 
getTrainPerf(m.rpart) #highest ROCL: 0.591, cp: 0.00, TPR:38% ,TNR: 72%
varImp(m.rpart) #most important var = admistype ret parole

#visualizing rpart 
plot(m.rpart) #the best ROC is low complexity
rpart.plot(m.rpart$finalModel) 

#predict rpart on test set
p.rpart <-  predict(m.rpart, x.test)
confusionMatrix(p.rpart, y.test) #acc 0.566, balacc 0.561

####naive bayes method####
set.seed(348)
modelLookup("nb")
m.nb <- train(y=y.train, x=x.train,
                 trControl = ctrl,
                 metric = "ROC", #using AUC to find best performing parameters
                 method = "nb")
m.nb
getTrainPerf(m.nb) #ROC: 0.592, TPR: 33%, TNR: 76%, chose this over the high TNR model bc it's more balanced
varImp(m.rpart) #most important var = admistype ret parole

#visualizing nb
plot(m.nb)

#predict nb on test set
p.nb <-  predict(m.nb, x.test)
confusionMatrix(p.nb, y.test) #acc 0.568, balacc 0.518

####logistic regression method####
set.seed(348)
modelLookup("glm")
m.glm <- train(y=y.train, x=x.train,
              trControl = ctrl,
              metric = "ROC", #using AUC to find best performing parameter
              method = "glm")
m.glm #ROC: 0.595, TPR: 55%, TNR: 99%
getTrainPerf(m.glm)
varImp(m.glm) #most important var = admis type other

#not visualzing bc there are no tuning parameters for glm

#predict glm on test set
p.glm <- predict(m.glm, x.test)
confusionMatrix(p.glm, y.test) #acc 0.570, balacc 0.520

####baging method####
set.seed(348)
modelLookup("treebag")
m.bag <- train(y=y.train, x=x.train,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameter
               method = "treebag")
m.bag #ROC: 0.573, TPR: 38%, TNR: 72%
getTrainPerf(m.bag)
varImp(m.bag) #return parole violator

#not visualizing bc there are no tunin paramters for bag

#predict bag on test set
p.bag <- predict(m.bag, x.test)
confusionMatrix(p.bag, y.test) #accuracy 0.566, bal acc 0.561

####random forest####
set.seed(348)
modelLookup("rf")
m.rf <- train(y=y.train, x=x.train,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameter
               tuneLength = 5, #5 to spare my poor computers processing power
               method = "rf")
m.bag 
getTrainPerf(m.rf) #ROC: 0.575, TPR: 47%, TNR: 65%
varImp(m.rf) #admis type other

#plot rf
plot(m.rf)

#predict bag on test set
p.bag <- predict(m.rf, x.test)
confusionMatrix(p.rf, y.test) #accuracy 0.566, bal acc 0.561

####boosting####
set.seed(348)
modelLookup("ada")
m.ada <- train(y=y.train, x=x.train,
              trControl = ctrl,
              metric = "ROC", #using AUC to find best performing parameter
              method = "ada")
m.ada 
getTrainPerf(m.ada) #ROC: 0.594, TPR: 23%, TNR: 83%
varImp(m.ada) #ethnicity white

#plot rf
plot(m.ada)

#predict bag on test set
p.ada <- predict(m.ada, x.test)
confusionMatrix(p.ada, y.test) #accuracy 0.569. balacc 0.518

####k nearest neighbor####
#getting error: "too many ties in knn"
#this likely means that there are too many neighbors equidistant to the target point, such that the algorithm cannot choose only k of them
set.seed(348)
modelLookup("knn")
m.knn <- train(y=y.train, x=x.train,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameter
               method = "knn")

m.knn <- knn(train=y.train, test=y.test, cl = y.train,
                k = 3)

####linear discriminant analysis####
set.seed(348)
modelLookup("lda")
m.lda <- train(y=y.train, x=x.train,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameter
               method = "lda")
m.lda
getTrainPerf(m.lda) #ROC: 0.595, TPR: 5%, TNR: 99%
varImp(m.lda) #ethnicity white

#not visualizing bc there are no tuning parameters for lda

#predict bag on test set
p.lda <- predict(m.lda, x.test)
confusionMatrix(p.lda, y.test) #accuracy 0.57. balacc 0.520

####Support Vector Machine####
set.seed(348) #started running 6:02pm
modelLookup("svmRadial")
m.svm <- train(y=y.train, x=x.train,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameter
               method = "svmRadial")
m.svm
getTrainPerf(m.svm) #ROC: 0.595, TPR: 5%, TNR: 99%
varImp(m.svm) #ethnicity white

#not visualizing bc there are no tuning parameters for lda

#predict bag on test set
p.lda <- predict(m.svm, x.test)
confusionMatrix(p.svm, y.test) #accuracy 0.57. balacc 0.520

#######################################
#Comparing training performance
####################################### 
#####create list of cross validation runs####
rValues <- resamples(list(rpart=m.rpart, naivebayes=m.nb, logistic=m.glm, bagging=m.bag, randomforest = m.rf, 
                          boosting = m.ada, LDA = m.lda, SVM = m.svm))
summary(rValues) #see average performance across the 10 folds

#compare models with visusals, TO DO GIVE TITLES
bwplot(rValues, metric="ROC") #Determine how high the variance is based on how wide the bars are. 
bwplot(rValues, metric="Sens") 
bwplot(rValues, metric="Spec") #thinking bagging is the best atm

#####AUC and roc curves comparisons####
rpart.prob <- predict(m.rpart, x.test, type = "prob")
nb.prob <- predict(m.nb, x.test, type="prob")
glm.prob <- predict(m.glm, x.test, type="prob")
bag.prob <- predict(m.bag, x.test, type="prob")
rf.prob <- predict(m.rf, x.test, type="prob")
ada.prob <- predict(m.ada, x.test, type="prob")
lda.prob <- predict(m.lda, x.test, type="prob")
lda.prob <- predict(m.svm, x.test, type="prob")

#TPR and FPR is calculated for continuum of probability threshold
rpart.roc<- roc(y.test, rpart.prob$max) 
nb.roc<- roc(y.test, nb.prob$max)
glm.roc<- roc(y.test, glm.prob$max)
bag.roc<- roc(y.test, bag.prob$max)
rf.roc<- roc(y.test, rf.prob$max)
ada.roc<- roc(y.test, ada.prob$max)
lda.roc<- roc(y.test, lda.prob$max)
lda.roc<- roc(y.test, svm.prob$max)

#lets see auc
auc(rpart.roc) #0.5836
auc(nb.roc) #0.5855
auc(glm.roc) #0.5869 #top
auc(bag.roc) #0.5678
auc(rf.roc) #0.5675
auc(ada.roc) #0.5853
auc(lda.roc) #0.5869 #top 
auc(svm.roc) #0.5869 #top 

#let's create an ROC plot with all combined, none are really great.
plot(rpart.roc, col="pink", xaxt= "n")
plot(nb.roc, add=T, col="purple", xaxt= "n")
plot(glm.roc, add=T, col="light blue", xaxt= "n")
plot(bag.roc, col="maroon", xaxt= "n")
plot(rf.roc, add=T, col="orange", xaxt= "n")
plot(ada.roc, add=T, col="dark blue", xaxt= "n")
plot(lda.roc, add=T, col="green", xaxt= "n")
plot(svm.roc, add=T, col="brown", xaxt= "n")
legend(x=.34, y=.3, cex=.5, 
       legend=c("rpart","Naive Bayes", "Logistic", "Bagging", "Random Forest", "Boosting", "LDA", "SVM"), 
       col=c("pink", "purple", " light blue", "maroon", "orange", "dark blue", "green", "brown"), lwd=5)
axis(1, at=x, labels=x, tck=0)
x <- c(1.0, 0.75, 0.5, 0.25, 0.0)


