## xgboost 101##
# code source1: https://www.kaggle.com/rtatman/machine-learning-with-xgboost-in-r/notebook
# code source2: https://www.r-bloggers.com/an-introduction-to-xgboost-r-package/ 
library(xgboost)
library(tidyverse)
library(skimr)

## use mushroom data set ##
#load data 
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test

# simple model
simplemodel <- xgboost(data = train$data, 
                 label = train$label,
                 nrounds = 2, # nrounds: the number of decision trees in the final model
                 objective = "binary:logistic") # objective: the training objective to use, specify the classifier
# [1]	train-error:0.000614 
# [2]	train-error:0.001228 

# make predictions
preds <- predict(simplemodel, test$data)

# cross-validation with xgb.cv
cv.res <- xgb.cv(data = train$data, 
                 label = train$label,
                 nfold = 5, # specifiy nfold
                 nrounds = 2,
                 objective = "binary:logistic")
# [1]	train-error:0.001075+0.000287	test-error:0.001228+0.000614 
# [2]	train-error:0.001228+0.000153	test-error:0.001228+0.000614 

## Kaggle tutorial ##
# load data 
diseaseInfo <- read.csv("/Users/lsy/OneDrive - UW-Madison/My R/my R repository/Outbreak_240817.csv")

## step1: shuffle data: ensure random samples in train/test dataset
set.seed(1234)
diseaseInfo <- diseaseInfo[sample(1:nrow(diseaseInfo)), ]

## step 2 data cleaning
## data cleaning for xgboost
## 1. Remove information about the target variable from the training data
## 2. Reduce the amount of redundant information
## 3. Convert categorical information (like country) to a numeric format
## 4. Split dataset into testing and training subsets
## 5. Convert the cleaned dataframe to a Dmatrix

# 1. remove information about the target variable from the training data: 
# knowing what the target label is supposed to be will certainly help us make accurate predictions 
# but a model that relies on the correct labels for classification won't be very helpful or interesting. 
diseaseInfo_humansRemoved <- diseaseInfo %>%
  select(-starts_with("human"))

# get a boolean vector of training labels
diseaseLabels <- diseaseInfo %>%
  select(humansAffected) %>% # get the column with the # of humans affected
  is.na() %>% # is it NA?
  magrittr::not() # switch TRUE and FALSE (using function from the magrittr package)

# 2. reduce the amount of redundant inforamtion
# select just the numeric columns
diseaseInfo_numeric <- diseaseInfo_humansRemoved %>%
  select(-Id) %>% # the case id shouldn't contain useful information
  select(-c(longitude, latitude)) %>% # location data is also in country data
  select_if(is.numeric) # select remaining numeric columns
# make sure that our dataframe is all numeric
str(diseaseInfo_numeric)

# 3. convert categorical information (like country) to a numeric format
# one-hot coding: a way to encode categorical variable into numeric 
# one-hot matrix for just the first few rows of the "country" column
# convert categorical factor into one-hot encoding
region <- model.matrix(~country-1,diseaseInfo)

# add a boolean column to our numeric dataframe indicating whether a species is domestic
diseaseInfo_numeric$is_domestic <- str_detect(diseaseInfo$speciesDescription, "domestic")

# get a list of all the species by getting the last
speciesList <- diseaseInfo$speciesDescription %>%
  str_replace("[[:punct:]]", "") %>% # remove punctuation (some rows have parentheses)
  str_extract("[a-z]*$") # extract the least word in each row

# convert our list into a dataframe...
speciesList <- tibble(species = speciesList)

# and convert to a matrix using 1 hot encoding
options(na.action='na.pass') # don't drop NA values!
species <- model.matrix(~species-1,speciesList)

# add our one-hot encoded variable and convert the dataframe into a matrix
diseaseInfo_numeric <- cbind(diseaseInfo_numeric, region, species)
diseaseInfo_matrix <- data.matrix(diseaseInfo_numeric)

# 4. split dataset into testing and training subsets
# 70% training, 30% testing
# get the numb 70/30 training test split
numberOfTrainingSamples <- round(length(diseaseLabels) * .7)

# training data
train_data <- diseaseInfo_matrix[1:numberOfTrainingSamples,]
train_labels <- diseaseLabels[1:numberOfTrainingSamples]

# testing data
test_data <- diseaseInfo_matrix[-(1:numberOfTrainingSamples),]
test_labels <- diseaseLabels[-(1:numberOfTrainingSamples)]

# 5. concert the cleaned df to a dmatrix (internal data structure of xgboost)
# optional, but dmatrix makes model train quicker
# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)

## step 3 Training the model

# model needs to know: 1) what training data to use; 2) the number of training rounds; 3) objective function
# building training model: train a model using our training data
model <- xgboost(data = dtrain, # the data   
                 nround = 2, # max number of boosting iterations
                 objective = "binary:logistic")  # the objective function
# [1]	train-error:0.015202 
# [2]	train-error:0.015202 
# same train error means no improvement in the 2nd round of training, so it's good to move to testing data

# testing model with testing set
# generate predictions for our held-out testing data
pred <- predict(model, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))
# [1] "test-error= 0.0121520972167777" 
# lower error on testing data than training data: didn't overfit!
# this is the basic model

## Step 4 Tuning model
# avoid overfitting (model relies too much on randomness/noise in training set): make the model less complex
# in xgboost: specifying the decision trees to have fewer layers rather than more layers
# bc each layer splits the remaining data into smaller and smaller pieces and therefore makes it more likely 
# the model capturing randomness and not the important variation
# default depths max.depth of trees in xgboost is 6

# train an xgboost model 1: max.depth = 3
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 2, # max number of boosting iterations
                       objective = "binary:logistic") # the objective function 
# [1]	train-error:0.015202 
# [2]	train-error:0.015202

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))
# [1] "test-error= 0.0121520972167777"

# no big changes in test-error in basic model and tuned model
# so how about under-fitting? two things to try
# thing 1: account for imbalances classes (weight it)
# thing 2: train for more rounds, but be careful bc it can also lead to over-fit.
# early_stopping_rounds to stop training if no improvement in a certain number of training rounds

# get the number of negative & positive cases in our data
negative_cases <- sum(train_labels == FALSE)
postive_cases <- sum(train_labels == TRUE)

# train a model using our training data
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 10, # number of boosting iterations
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       scale_pos_weight = negative_cases/postive_cases) # control for imbalanced classes
# [1]	train-error:0.015118 
# Will train until train_error hasn't improved in 3 rounds.
# [2]	train-error:0.015118 
# [3]	train-error:0.015118 
# [4]	train-error:0.015118 
# Stopping. Best iteration:
# [1]	train-error:0.015118

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))
# [1] "test-error= 0.0121520972167777"

# train an xgboost model 2: with a regularization term gamma (default gamma = 0)
# adding a regularization term makes the model more conservativeGamma is a measure of 
# how much an additional split will need to reduce loss in order to be added to the ensemble. 
# If a proposed model does not reduce loss by at least whatever-you-set-gamma-to, it won't be included.
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 10, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term
# [1]	train-error:0.015118 
# Will train until train_error hasn't improved in 3 rounds.
# [2]	train-error:0.015118 
# [3]	train-error:0.015118 
# [4]	train-error:0.015118 
# Stopping. Best iteration:
# [1]	train-error:0.015118

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))
# [1] "test-error= 0.0121520972167777"


## Step 5 Examine the model

# find the most contributing features
# plot them features: find the contributing most to the model (hyperparameter)
xgb.plot.multi.trees(feature_names = names(diseaseInfo_matrix), 
                     model = model)

# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}

# probability of leaf above countryPortugul
odds_to_probs(-0.599)

# importance matrix: how important each feature is 
# get information on how important each feature is
importance_matrix <- xgb.importance(names(diseaseInfo_matrix), model = model)

# plot imporance matrix
xgb.plot.importance(importance_matrix)




