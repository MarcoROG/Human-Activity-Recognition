# Human-Activity-Recognition
Machine learning algorithm for HAR

##WriteUp
You can find the writeup of the code [HERE](http://htmlpreview.github.io/?https://github.com/MarcoROG/Human-Activity-Recognition/blob/master/Report.html)  
Note that you can't see the command outputs but only the markdown in the section shown on the readme, so for the full document click on "HERE" at the beginning of this section

##Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

##Data
The training data for this project are available here:   
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
The test data are available here:   
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

##Loading data
```{r}
library(caret)
#Set a seed for coherent tests
set.seed(12345)
training = read.csv("Data/pml-training.csv", header = TRUE)
finalTest = read.csv("Data/pml-testing.csv", header = TRUE)
```
##Data Preprocessing
We split the preprocessing in two phases: a generic preprocessing and a trainingSet-specific preprocessing. 

####Generic preprocessing
In generic preprocessing we drop useless columns, we factorize the ```classe``` variable, and we set all the missing values to NA.
```{r}
toDrop = c("user_name","X","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window")
training = training [,!(names(training) %in% toDrop) ]
finalTest = finalTest [,!(names(finalTest) %in% toDrop)]

#Convert training-set's "Classe" column to factor variable
training[,"classe"] = as.factor(training[,"classe"])

#Set null or empty values to NA
is.na(training[,]) <- training[,names(training)] == ""
is.na(training[,]) <- training[,names(training)] == "NULL"
```

####Data splitting
We proceed by splitting the data in a training and test set:
```{r}
#Split the training set with 60-40
trainIndexes <- createDataPartition(training$classe, p = .6,list = FALSE)
#Remove "classe" since this should be treated as a test set, so without labels
test <- training[-trainIndexes,]
training <- training[trainIndexes,]
#Clean RAM
rm(trainIndexes)
```

####TrainingSet-specific preprocessing
Here we do all the preprocessing operations that depend on parameters that must be determined on the training set only, and then we apply them to the other sets aswell:  
We remove all the columns that have more than 50% of null values, since they could give problems with knnImputees  
Then we create a preprocess object that applies knnImputees to fill in missing data, plus centering and scaling for algorithm efficiency.
Both PCA and NZV were considered in this case, but they weren't so usefull because of weak correlations and generally high variance between values.  
```{r}
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


```
####Validation set
We then process the test sets with the same procedure and then create a further split in the test data, to obtain a validation set.  
This hasn't been before for the ease of applying the same preprocessing operation to only one set.  
We keep 75% in test set cause we need it to train our combining predictor.
```{r}
testIndexes <- createDataPartition(test$classe, p = .75,list = FALSE) 
validation <- test[-testIndexes,]
test <- test[testIndexes,]
rm(testIndexes)
```
##Model fitting
First of all, we declare a ```{r}trainControl``` object which will take care of repeating 5-fold CV 3 times in order to tune parameters.  
Then we go on and train three models on the training set. We deceided to use multiple models for accademic purposes, since multi-model prediction was in the scope of the course.  
The three algorithms are:  
-Penalized Multinomial Regression neural network, since NN represent the state of the art in deep learning techniques  
-SVM (Support Vector Machine) which is one of the most recent and powerfull algorithms and LMC (Large Margin Classifier)  
-Random Forest, for its versatility and ability to perform feature extraction, seen extensively in the course
```{r, message="False", results="hide"}
#Fit a NN algorithm
NN <- train(training$classe ~ ., data=as.data.frame(training), method="multinom",trainControl=trainControl, allowParallel=TRUE, verbose=FALSE)
#Fit an SVM, one of the most advanced algorithms
SVM <- train (training$classe ~ .,data=training,method="svmLinear",trainControl=trainControl,allowParallel=TRUE, verbose=FALSE)
#Fit a random forest, for its ability to detect usefull features and correlations
RF <- train(training$classe ~ ., data=training, method="rf", trainControl=trainControl,allowParallel=TRUE, verbose=FALSE)
```
##Model evalutation
Here we use the models to predict the test set data, and then the ConfusionMatrix to evalutate their accuracy:
```{r}
SVMpredict <- predict(SVM, newdata=test)
SVMaccuracy <- confusionMatrix(SVMpredict,test$classe)$overall['Accuracy']
SVMaccuracy

RFpredict <- predict(RF, newdata=test)
RFaccuracy <- confusionMatrix(RFpredict,test$classe)$overall['Accuracy']
RFaccuracy

NNpredict <- predict(NN, newdata=test)
NNaccuracy <- confusionMatrix(NNpredict,test$classe)$overall['Accuracy']
NNaccuracy
```
##Combining models
In this section we use the models predictions on the test set, and the test set labels, to train a NaiveBayes classifier that can predict a result class based on the other three models predictions.  
This is usefull because the combined model can learn to take into account intrinsic errors in the fed-in data.  
```{r}
#Combine the predictors
combinedPredictors <- data.frame(SVMpredict,RFpredict,NNpredict,test$classe)
#Train a predictor
combinedModel <- train(combinedPredictors$test.classe ~ ., data= combinedPredictors, method = "nb")
```
##Evalutating the combined model
Here we use the three models to predict the labels for the validation set, then we frame those prediction and feed them to the combined model in order to evaluate its accuracy in an out-of-sample set:
```{r}
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
```

##Predicting the final labels
Here we use the three models plus the combined model to predict the result of the ```{r}finalTest``` set, which is the one provided for the assignment. 
```{r}
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
```

##Conclusion
The combined model accuracy is slightly lower than the RandomForest one.  
As i said, i went for a multimodel classifier just for didactical purposes.  
RandomForest proves again to be a great algorithm at the expense of very long training times, with more than 99% accuracy out-of-sample.  
In the end, all the results predicted by the algorithm are correct, and the final prediction is really fast aswell.  
Some future work may include some optimization, parameters tuning and maybe a deeper experimentation in various types of models.  
It is also reccomended to port this algorithm in another programming language for practical purposes.

###Clarification on style
You may notice I use the "we" pronoun when i think of the choiches made for this algorithm, this is something i'm used to.  
All the work is done solely by me, Marco Bellan, using notions acquired from the Practical Machine Learning Course, along with the R and Caret wiki and documentation.
