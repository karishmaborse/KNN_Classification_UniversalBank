# K-Nearest Neighbors (K-NN)

# Importing the dataset
dataset = read.csv('UniversalBank.csv')
dataset = dataset[3:5]
#Selecting the predictors for classification
Selectvariable <- c(2,3,4,6,7,8,9,10,11,12,13,14)
#Importing to dataset
ubdf <- read.csv("UniversalBank.csv")
set.seed(1)
#Training and test dataset sampled
trainindex <- sample(row.names(ubdf), 0.6*dim(ubdf)[1]) ## training data - 60% of the records
testindex <- setdiff(row.names(ubdf), trainindex)
traindf <- ubdf[trainindex, Selectvariable]
testdf <- ubdf[testindex, Selectvariable] #test data set
# initialize normalized training, validation data, complete data frames to originals
trainnormdf <- traindf
tesetnormdf <- testdf
ubdfnormal <- ubdf
# use preProcess() 
install.packages("caret")
library(caret)
normvalues <- preProcess(traindf[, 1:12], method=c("center", "scale"))
trainnormdf[, -8] <- predict(normvalues, traindf[, -8])
tesetnormdf[, -8] <- predict(normvalues, testdf[, -8])
ubdfnormal[, -8] <- predict(normvalues, ubdf[, -8])
# use knn() to compute knn.
# Feature Scaling of training and testing sets
trainnormdf[-8] = scale(traindf[-8])
tesetnormdf[-8] = scale(testdf[-8])

# use knn() to compute knn.
# knn() is available in library FNN (provides a list of the nearest neighbors)
# and library class (allows a numerical output variable).
library(class)
dftrain <- as.data.frame(trainnormdf)
dftest <- as.data.frame(tesetnormdf)

y_pred <- knn(train = dftrain[,-8],
              test = dftest[,-8],
              cl = trainnormdf$Personal.Loan, k = 1)

#Confusion matrix for testset
cm = table(dftest[, 8],y_pred)

accuracy = sum(diag(cm))/length(dftest[,8])


# for new record
new_df = list(Age=40.0, Experience=10.0, Income=84.0,  Family=2, CCAvg=2, Education=2, Mortgage=0, Personal.Loan=0, Securities.Account=0, CD.Account=0, Online=1, CreditCard=1)
new_df = as.data.frame(new_df)
newnormdf<-new_df
newnormdf[, -8] <- predict(normvalues, new_df[, -8])


y_pred_new <- knn(train = dftrain[,-8],
                  test = newnormdf[,-8],
                  cl = dftrain$Personal.Loan, k = 1)
print(y_pred_new) # prdicted value is 0 for this new data



# finding the best k value
for(i in 1:14) {
  y_pred <- knn(train = dftrain[,-8],
                test = dftest[,-8],
                cl = trainnormdf$Personal.Loan, k = i)
  cm = table(dftest[, 8],y_pred)
  accuracy = sum(diag(cm))/length(dftest[,8])
  cat("Accuracy for k value " , i , "is ", accuracy , "\n")
}


# for new record, when k=3
new_df = list(Age=40.0, Experience=10.0, Income=84.0,  Family=2, CCAvg=2, Education=2, Mortgage=0, Personal.Loan=0, Securities.Account=0, CD.Account=0, Online=1, CreditCard=1)
new_df = as.data.frame(new_df)
newnormdf<-new_df
newnormdf[, -8] <- predict(normvalues, new_df[, -8])


y_pred_new <- knn(train = dftrain[,-8],
                  test = newnormdf[,-8],
                  cl = dftrain$Personal.Loan, k = 3)
print(y_pred_new) # prdicted value is 0 for this new data
cm_new = table(newnormdf[, 8],y_pred_new)
accuracy_new = sum(diag(cm_new))/length(newnormdf[,8])

cat("Accuracy for k=3 is ", accuracy_new , "\n")

#It is overfitting






