#Time series forecasting using Machine Learning: An example from Kaggle

#Contest Description: Corporación Favorita, a large Ecuadorian-based grocery retailer that operates hundreds
#of supermarkets, with over 200,000 different products on their shelves, has challenged the Kaggle community 
#to build a model that more accurately forecasts product sales. They currently rely on subjective forecasting 
#methods with very little data to back them up and very little automation to execute plans. 
#They’re excited to see how machine learning could better ensure they please customers by having just enough of 
#the right products at the right time.

#Corporacion Favorita provided historical sales data from 2012 - August 15th, 2017. The challenge is to predict sales 
#from August 16th - August 31st 2017. 

#Time series analysis techniques such as ARIMA, moving averages and lag variables along with seasonal trends are often
#used to forecaset for these types of problems. This tutorial will discuss how to combine these techniques with 
#machine learning to build more accurate predictive models.

#Steps to building the model
#1. Data Cleansing
#2. Feature Engineering
#3. Feature Selection
#4. Formatting data for the machine learning model
#5. Build model(s)
#6. Test model(s)
#7. Model Parameter Tuning

#Note: Exploratory data analysis is generally completing early in a machine learning project. 
#      For this project, I didn't do much because the Kaggle community graciously did this part of the project and
#      shared with the world.

#

library(data.table)
library(dplyr)
library(padr)
library(xgboost)
library(Matrix)
library(RcppRoll)
library(zoo)

########################## Data Cleansing ##########################################
# train <- fread('./input/train.csv')
# 
# train$date <- as.Date(train$date)
# 
# #Select years to use. Left out 2012.
# Years <- c(2013:2017)
# 
# #In the test dataset there are entries for items which had 0 sales for a given day. The training dataset only contains
# #products on days where there was a sale. The training dataset needs to include the products where no sales occured
# # on a given day. I use the padr package to add in these missing zeros.
# 
# #<Show examples of old and new data>
# 
# for (i in 1:length(Years)){
#   MinYear <- paste0(Years[i], '-02-01')
#   MaxYear <- paste0(Years[i], '-09-01')
#   df <- subset(train, date >= MinYear & date < MaxYear)
#   df <- pad(df, start_val = as.Date(MinYear)
#                , end_val = as.Date(MaxYear), interval = 'day'
#                , group = c('store_nbr', 'item_nbr'), break_above = 100000000000
#   ) %>%
#     fill_by_value(unit_sales)
# 
#   write.csv(df, paste0('train_2', Years[i], '.csv'))
# }
# 
# #Probably not necessary
# Years <- c(2013:2017)
# df <- data.frame()
# for (i in 1:length(Years)){
#   x <- fread(paste0('train_2', Years[i], '.csv'))
#   df <- rbind(df, x)
# 
# }
# 
# #Remove
# df$V1 <- NULL
# 
# #Remove negative sales day.
# df$unit_sales[df$unit_sales < 0] <- 0
# 
# #Added some unnecessary zeros to the year 2017. Move this up in the code to where it's needed. Just after the dataset is created.
# df <- df[df$date < '2017-08-16',]
# 
# #Convert to log of sales. The evaluation metric of the competition is the Normalized Weighted Root Mean Squared Logarithmic Error.
# #If you're interested, you can read more about this at the competitions webpage.
# df$unit_sales <- log1p(df$unit_sales)
# # #
# # #
# #Read in test dataset
# test <- fread('./input/test.csv')
# #
# #Bind together. Fill is set to TRUE so that the unit sales column is filled with NAs in the test dataset. They don't have
# #the same columns
# df <- rbind(df, test, fill = TRUE)
# #
# # #############################Feature Engineering###################################
# 
# #Create and Modify variables
# df$month <- as.numeric(substr(df$date, 6, 7))
# df$day <- as.numeric(as.factor(weekdays(as.Date(df$date))))
# df$day_num <- as.numeric(substr(df$date, 9, 10))
# 
# 
# df$onpromotion <- as.numeric(df$onpromotion)
# 
# #Not sure if necessary or beneficial
# df$onpromotion[is.na(df$onpromotion)] <- 0
# #
# #Read in other data sets
# items <- fread('./input/items.csv')
# stores <- fread('./input/stores.csv')
# transactions <- fread('./input/transactions.csv')
# holidays <- fread('./input/holidays_events.csv')
# 
# df <- merge(df, items, by = 'item_nbr')
# 
# #Set categorical variables to numeric. Necessary for input into the machine learning models.
# df$family <- as.numeric(as.factor(df$family))
# 
# df <- merge(df, stores, by = 'store_nbr')
# df$type <- as.numeric(as.factor(df$type))
# #
# #Calculate the average transactions per store
# transactions_agg <- setNames(aggregate(transactions$transactions, by = list(transactions$store_nbr), FUN = mean)
#                              , c('store_nbr', 'transaction_avg'))
# df <- merge(df, transactions_agg, by = 'store_nbr')
# 
# #Move up into previous section
# oil <- fread('./input/oil.csv')
# 
# # #Add NAs to the weekend dates that are missing
# oil$date <- as.Date(oil$date)
# oil <- pad(oil, start_val = as.Date(min(oil$date))
#            , end_val = as.Date(max(oil$date)), interval = 'day'
#            , break_above = 100000000000
# ) %>%
#   fill_by_value(dcoilwtico, value = NA)
# #
# # #Convert NAs to previous value. Show example
# oil$dcoilwtico <- na.locf(oil$dcoilwtico)
# #
# # #Set to character. More easy to subset data
# oil$date <- as.character(oil$date)
# #
# df <- merge(df, oil, by = 'date')
# #
# #
# write.csv(df, 'df_cleaned_2.csv', row.names = FALSE)
#####################

df <- fread('df_cleaned_2.csv')

df$city <- as.numeric(as.factor(df$city))
df$year <- substr(df$date, 1, 4)

train <- subset(df, date < '2017-08-16')

#Order by date to create the lag and moving average variables
train <- train[order(train$date),]

#Here I create new features using a variety of statistical summary stats moving average and lag variables grouped by store_nbr and item_nbr using the RcppRoll package. 

#<Show example of what's happening>
train <- train %>%
  group_by(store_nbr, item_nbr) %>%
  mutate(avg_7 = lag(roll_meanr(unit_sales, 7), 1)
         , avg_30 = lag(roll_meanr(unit_sales, 30), 1)
         , avg_60 = lag(roll_meanr(unit_sales, 60), 1)
         , avg_89 = lag(roll_meanr(unit_sales, 89), 1)
         , avg_140 = lag(roll_meanr(unit_sales, 140), 1)
         # , min_30 = lag(roll_minr(unit_sales, 30), 1)
         # , max_30 = lag(roll_maxr(unit_sales, 30), 1)
         # , med_31 = lag(roll_medianr(unit_sales, 31), 1)
         ) %>%
  #mutate(lag_1 = lag(unit_sales, 1)) %>%
  mutate(lag_promo = lag(onpromotion, 1)) %>%
  mutate(lag_promo14 = lag(roll_sumr(onpromotion, 14), 1)) %>%
  mutate(lag_promo30 = lag(roll_sumr(onpromotion, 30), 1)) %>%
  mutate(lag_promo89 = lag(roll_sumr(onpromotion, 89), 1))

train <- subset(train, month > 6)
train2013 <- train[train$month > 7 & train$year == 2013,]
train2014 <- train[train$month > 7 & train$year == 2014,]
train2015 <- train[train$month > 7 & train$year == 2015,]
train2016 <- train[train$month > 7 & train$year == 2016,]
train2017 <- train[train$month > 6 & train$year == 2017,]
train <- rbind(train2013, train2014, train2015, train2016, train2017)
rm(train2013, train2014, train2015, train2016, train2017)

#Probably not necessary
train <- train[!is.na(train$avg_140),]

label <- train$unit_sales

#Probably not necessary. Should check for NAs prior to this
previous_na_action<- options('na.action')
options(na.action='na.pass')

#Build matrix input for the model
trainMatrix <- sparse.model.matrix(~ avg_7 + avg_60 + avg_30 + avg_89 + avg_140 +
                                     dcoilwtico + day + day_num + family + class + city +
                                     onpromotion + type + cluster + lag_promo14 + lag_promo + lag_promo30 +  lag_promo89 +
                                     transaction_avg
                                   , data = train
                                   , contrasts.arg = c('day', 'day_num', 'family', 'class', 'city'
                                                       , 'type', 'cluster')
                                   , sparse = FALSE, sci = FALSE)

options(na.action = previous_na_action$na.action)



#Create input for xgboost
trainDMatrix <- xgb.DMatrix(data = trainMatrix, label = label)

#Remove data from in-memory so I don't run out.
rm(train)
gc()

#Set parameters of model
params <- list(booster = "gbtree"
               , objective = "reg:linear"
               , eta=0.4
               , gamma=0
               , max_depth=8
               , min_child_weight=4
               , subsample=.8
               , colsample_bytree=.7)


#Cross-validation
# xgb.tab <- xgb.cv(data = trainDMatrix
#                   , param = params
#                   , maximize = FALSE, evaluation = "rmse", nrounds = 500
#                   , nthreads = 10, nfold = 2, early_stopping_round = 10)

model_list <- list()

#Number of rounds
min.error.idx = 100#xgb.tab$best_iteration

#Number of models to build
numb = 3
i = 1

#Loop to build list of models. Could include different types of models as well. Neural net, regression, random forest
for (i in 1:numb){
  model_list[[i]] <- xgb.train(data = trainDMatrix
                               , param = params
                               , maximize = FALSE, evaluation = 'rmse', nrounds = min.error.idx)
  }


rm(trainMatrix)
gc()

#Dates vector for the testing dataset
Dates <- seq.Date(from = as.Date('2017-08-16'), by = 'day', to = as.Date('2017-08-31' ))

#Loop to predict sales day-by-day for the test dataset. It predicts for the first day in the test dataset (8/16). Then inputs
#that prediction in the testing dataset and predicts for the next day (8/17) and so on. One problem with this approach
#is any errors in predictions are propagated through the model. This can decrease the accuracy of the longer-term predictions.
df_test <- df
df <- NULL
gc()
i=1
for (i in 1:length(Dates)){
  gc()
  #Order test dataset
  df_test <- df_test[order(df_test$date),]
  
  #Create lag variables on the testing data
  df_test <- df_test %>%
    group_by(store_nbr, item_nbr) %>%
    mutate(avg_7 = lag(roll_meanr(unit_sales, 7), 1)
           , avg_30 = lag(roll_meanr(unit_sales, 30), 1)
           , avg_60 = lag(roll_meanr(unit_sales, 60), 1)
           , avg_89 = lag(roll_meanr(unit_sales, 89), 1)
           , avg_140 = lag(roll_meanr(unit_sales, 140), 1)
           # , min_30 = lag(roll_minr(unit_sales, 30), 1)
           # , max_30 = lag(roll_maxr(unit_sales, 30), 1)
           # , med_31 = lag(roll_medianr(unit_sales, 31), 1)
    ) %>%
    #mutate(lag_1 = lag(unit_sales, 1)) %>%
    mutate(lag_promo = lag(onpromotion, 1)) %>%
    mutate(lag_promo14 = lag(roll_sumr(onpromotion, 14), 1)) %>%
    mutate(lag_promo30 = lag(roll_sumr(onpromotion, 30), 1)) %>%
    mutate(lag_promo89 = lag(roll_sumr(onpromotion, 89), 1))
  
  #Subset testing data to predict only 1 day at a time
  test <- df_test[df_test$date == as.character(Dates[i]),]
  #Remove NAs
  test <- test[!is.na(test$id),]
  test <- test[!is.na(test$avg_140),]
  
  
  previous_na_action<- options('na.action')
  options(na.action='na.pass')
  
  testMatrix <- sparse.model.matrix(~ avg_7 + avg_60 + avg_30 + avg_89 + avg_140 +
                                    dcoilwtico + day + day_num + family + class + city +
                                      onpromotion + type + cluster + lag_promo14 + lag_promo + lag_promo30 +  lag_promo89 +
                                      transaction_avg
                                    , data = test
                                    , contrasts.arg = c('day', 'day_num', 'family', 'class', 'city'
                                                        , 'type', 'cluster')
                                    , sparse = FALSE, sci = FALSE)
  
  
  options(na.action = previous_na_action$na.action)
  
  #Predict values for a given day
  #Loop
  pred_list <- list()
  j=1
  for(j in 1: length(model_list)){
    pred_list[[j]] <- predict(model_list[[j]], testMatrix)
  }
  pred_list <- data.frame(pred_list)
  
  #Mean of the predictions
  pred <- apply(pred_list, 1, FUN = mean)

  pred[pred < 0] <- 0

  #Add predicted values to the dataset.
  df_test$unit_sales[df_test$id %in% test$id] <- pred
  print(i)
  
}
#Only include the testing dates
df_test <- df_test[df_test$date >= '2017-08-16',]

#Set NA values to zero
df_test$unit_sales[is.na(df_test$unit_sales)] <- 0

#Convert predictions back to normal unit from the log.
df_test$unit_sales <- expm1(df_test$unit_sales)

#Create solution dataset
solution <- df_test[,c('id', 'unit_sales')]

#Write solution csv
write.csv(solution, 'final_29.csv', row.names = FALSE)
