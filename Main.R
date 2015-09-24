library(caret)
#Set a seed for coherent tests
set.seed(12345)

#Load datasets
training = read.csv("Data/pml-training.csv", header = TRUE)
finalTest = read.csv("Data/pml-testing.csv", header = TRUE)

##########################################################################################
#########################          GENERIC PREPROCESSING         #########################
##########################################################################################
#> Remove some useless predictors
#> Factorize 'Classe'
#> Set NULL or Empty values to NA

#Remove useless columns which can only lead to overfitting
#The time of the day in which an exercise is performed cannot influence the quality of its execution
#X is the order in which the sample was measured, this should not influence the quality either
#The name will not influence it either along with window information
toDrop = c("user_name","X","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window")
training = training [,!(names(training) %in% toDrop) ]
finalTest = finalTest [,!(names(finalTest) %in% toDrop)]

#Convert training-set's "Classe" column to factor variable
training[,"classe"] = as.factor(training[,"classe"])

#Set null or empty values to NA
is.na(training[,]) <- training[,names(training)] == ""
is.na(training[,]) <- training[,names(training)] == "NULL"

##########################################################################################
#########################          DATA PARTITIONING         #############################
##########################################################################################
#>Split datasets in order to have a test, a training and a finalTest sets

#Split the training set with 60-40
trainIndexes <- createDataPartition(training$classe, p = .6,list = FALSE)
#Remove "classe" since this should be treated as a test set, so without labels
test <- training[-trainIndexes,]
training <- training[trainIndexes,]
#Clean RAM
rm(trainIndexes)

##########################################################################################
#################          TrainingSet-Specific PREPROCESSING         ####################
##########################################################################################
#>Remove columns with too many NA values (this would cause imperfect knn) from all the sets
# but perform the analisys only on the training set, in order to avoid fitting our model in the
# evalutation sets aswell.

#Find all the columns with less than 50% NA values
training <- training[,(colSums(is.na(training[,names(training)])) / nrow(training) ) < 0.5]
#Save this for later removing the same vars from the test set aswell
toKeep <- names(training)
test <- test [, toKeep]
toKeepTest <- toKeep[toKeep !="classe"]
#Keep the same variables, since we have to always use the same model
finalTest <- finalTest[,toKeepTest]


#Create a preprocessing object using the training set
#PCA and NZV were considered but they wouldn't be usefull in this case

#Extract everything except the solution
tr <- training[,!(colnames(training)=="classe")] 
#Preprocess
preprocessing <- preProcess(tr,method=c("knnImpute","center","scale"))
#Save the solutions for the training set
classe <- training$classe
#Clear memory
rm(training)
#Reassign the training set with the labels
training <- predict(preprocessing,newdata=tr)
training$classe <- classe
#Remove the temporary variables
rm(tr)
rm(classe)

#>Preprocess the test sets
te <- test[,!(colnames(test)=="classe")]
classe <- test$classe
rm(test)
test <- predict(preprocessing,newdata=te)
test$classe <- classe

finalTest <- predict(preprocessing,newdata=finalTest)

#Remove the temporary variables
rm(te)
rm(classe)

#Create a validation set, it's done here so that we can avoid doing the preprocessing one more time
testIndexes <- createDataPartition(test$classe, p = .75,list = FALSE) #Keep 75% in test set cause we need it to train our combining predictor
validation <- test[-testIndexes,]
test <- test[testIndexes,]
rm(testIndexes)

##########################################################################################
############################          MODEL FITTING         ##############################
##########################################################################################

#Training controller
trainControl <- trainControl(method = "repeatedcv", number = 5,  repeats = 3)

#We're going to fit multiple models, maybe just a RF model would have been enough, but this is a nice practice
#for predicting with more than one model, which was in the scope of the course.

#Fit a NN algorithm
NN <- train(training$classe ~ ., data=as.data.frame(training), method="multinom",trainControl=trainControl, allowParallel=TRUE, verbose=FALSE)
#Fit an SVM, one of the most advanced algorithms
SVM <- train (training$classe ~ .,data=training,method="svmLinear",trainControl=trainControl,allowParallel=TRUE, verbose=FALSE)
#Fit a random forest, for its ability to detect usefull features and correlations
RF <- train(training$classe ~ ., data=training, method="rf", trainControl=trainControl,allowParallel=TRUE, verbose=FALSE)

##########################################################################################
##########################          MODEL EVALUTATION         ############################
##########################################################################################
SVMpredict <- predict(SVM, newdata=test)
SVMaccuracy <- confusionMatrix(SVMpredict,test$classe)$overall['Accuracy']
SVMaccuracy

RFpredict <- predict(RF, newdata=test)
RFaccuracy <- confusionMatrix(RFpredict,test$classe)$overall['Accuracy']
RFaccuracy

NNpredict <- predict(NN, newdata=test)
NNaccuracy <- confusionMatrix(NNpredict,test$classe)$overall['Accuracy']
NNaccuracy

##########################################################################################
##########################          MODEL COMBINING         ##############################
##########################################################################################

#Combine the predictors
combinedPredictors <- data.frame(SVMpredict,RFpredict,NNpredict,test$classe)
#Train a predictor
combinedModel <- train(combinedPredictors$test.classe ~ ., data= combinedPredictors, method = "nb")

#Predict for validation
SVMpredict <- predict(SVM, newdata= validation)
RFpredict <- predict(RF, newdata= validation)
NNpredict <- predict(NN, newdata= validation)

#Combine the data
combinedPredictors <- data.frame(SVMpredict,RFpredict,NNpredict)
combinedPredictors$test.classe <- validation$classe

#Make it predict for evalutation
combinedPrediction <- predict(combinedModel,newdata=combinedPredictors)
combinedAccuracy <- confusionMatrix(combinedPrediction,validation$classe)$overall['Accuracy']
combinedAccuracy

##########################################################################################
##########################          FINAL PREDICTION         #############################
##########################################################################################

rm(SVMpredict)
rm(RFpredict)
rm(NNpredict)

#Measure time to see how long it takes
startTime<- proc.time()

SVMpredict <- predict(SVM, newdata=finalTest)
RFpredict <- predict(RF, newdata= finalTest)
NNpredict <- predict(NN, newdata= finalTest)

finalCombinedData<- data.frame(SVMpredict,RFpredict,NNpredict)

outCome <- predict(combinedModel, finalCombinedData)

proc.time() - startTime

outCome

##########################################################################################
#############################          SAVE FILES         ################################
##########################################################################################

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("Data/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

#pml_write_files(outCome)