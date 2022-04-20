library(tidyverse)
library(keras)
library(glmnet)
library(rpart) 
library(ranger)
library(caret)


train <- read_csv("../AS3/ceu-ml-2022/train.csv")
test <- read_csv("../AS3/ceu-ml-2022/test.csv")

my_seed = 20220412

# train <- train %>% mutate(
#   is_popular = as.factor(is_popular)
# )

# EDA ----

skimr::skim(train)
cor(train[1:60], train$is_popular)

ggplot(gather(train, cols, value), aes(value)) +
  geom_histogram() +
  facet_wrap(.~cols, scales = "free")

# feature engineering ----

# removing outlier values
train <- train %>% filter(n_unique_tokens <= 1)

# variables to factor
factors <- train %>% select(contains("data_channel")|contains("week")) %>% names()
train[factors] <- lapply(train[factors], factor)
# train$is_popular <- factor(train$is_popular, level = c(1,0))


# sapply(train, class)


# # split train set
# n_obs <- nrow(data)
# test_share <- 0.2
# 
# set.seed(20220411)
# test_indices <- sample(seq(n_obs), floor(test_share * n_obs))
# data_test <- slice(data, test_indices)
# data_train <- slice(data, -test_indices)

# function to submit to kaggle ----

# submitToKaggle <- function(model) {
#   pred <- select(test, article_id) %>% 
#     mutate(score = predict(model, newx = model.matrix(~., test[predictors]), type = "class", s = model$lambda.min)[,1]) 
#     write.csv(pred, paste0("../AS3/models/", deparse(substitute(model)), ".csv"), row.names = F)
# }


# LASSO -----

# set the predictor and response columns
response <- "is_popular"
predictors <- setdiff(names(train), c(response, "article_id"))


features <- model.matrix(~., train[predictors])
outcome <- train$is_popular

lasso <- cv.glmnet(features,  outcome, alpha = 1)

plot(lasso)
lasso
lasso$lambda.min

submitToKaggle(model = lasso)

lasso_pred <- predict(lasso, newx = model.matrix(~., test[predictors]), type = "class", s = lasso$lambda.min)



# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = lasso_pred[,1]) %>% 
  write.csv("../AS3/models/lasso.csv", row.names = F)
  

# PCA ------ doesnt w when outcome is factor !! ----
library(pls) 

pcr_model <- pcr(outcome ~ features[,-1], scale = TRUE)
summary(pcr_model)

save(pcr_model, file = "/Users/oszkar/Documents/CEU/Winter/DS/AS3/models/pca.RData")
pca_pred <- predict(pcr_model, newdata = as.matrix(test[, predictors]), ncomp = 59)
View(as.data.frame(pca_pred))

# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = pca_pred) %>% 
  write.csv("../AS3/models/pca.csv", row.names = F)


# Tree ----

tree_model <- rpart(
  formula(paste0("is_popular ~ ", paste(predictors, collapse = "+"))), 
  train
  )


tree_model$y

# variables to factor in test set -- doesn't work w factor outcome var
test[factors] <- lapply(test[factors], factor)

tree_pred <- predict(tree_model, newdata = test, type = 'prob')

save(tree_pred, file = "/Users/oszkar/Documents/CEU/Winter/DS/AS3/models/tree.RData")

View(tree_pred)

data.frame(tree_pred)
as.array(tree_pred)


# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = tree_pred) %>% 
  write.csv("../AS3/models/tree.csv", row.names = F)

# RF ---- factor outcome doesnt work ----

set.seed(my_seed)
simple_rf <- ranger(
  formula(paste0("is_popular ~ ", paste(predictors, collapse = "+"))), 
  train
)

save(simple_rf, file = "../AS3/models/rf.RData")
rf_pred <- predict(simple_rf, test)

# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = rf_pred$predictions) %>% 
  write.csv("../AS3/models/rf.csv", row.names = F)  

# GBM ----
library(gbm)
gbm <- gbm(
  formula(paste0("is_popular ~ ", paste(predictors, collapse = "+"))),
  data = train,
  n.trees = 5000,
  distribution = "gaussian",
  cv.folds = 5,
  shrinkage = 0.1,
  interaction.depth = 6
)
save(gbm, file = "../AS3/models/gbm.RData")

gbm_pred <- predict(gbm, test)

# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = gbm_pred) %>% 
  write.csv("../AS3/kaggle_submission_gbm.csv", row.names = F) 


# XGBoost ----
library(h2o)
h2o.init()

h2o <- as.h2o(train)
validation_h2o <- as.h2o(test)

h2o_data_splits <- h2o.splitFrame(data =  h2o, ratios = 0.8, seed = my_seed)
train_h2o <- h2o_data_splits[[1]]
test_h2o <- h2o_data_splits[[2]]

simple_xgboost <- h2o.xgboost(
  x = predictors, y = response,
  model_id = "simple_xgboost",
  training_frame = train_h2o,
  validation_frame = test_h2o,
  nfolds = 5,
  max_depth = 2, min_split_improvement = 0.1,
  learn_rate = 0.05, ntrees = 1000,
  score_each_iteration = TRUE,
  seed = 20220412
)
simple_xgboost

h2o.performance(simple_xgboost, xval = T)

xgb_pred <- predict(simple_xgboost, validation_h2o, type = 'prob')
xgb_pred <- as.data.frame(xgb_pred)

# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = xgb_pred[,1]) %>% 
  write.csv("../AS3/models/xgboost2.csv", row.names = F) 

save(simple_xgboost, file = "../AS3/models/xgboost2.RData")


# Auto ML ----

automl <- h2o.automl(
  y = response,
  training_frame = train_h2o,
  validation_frame = test_h2o,
  nfolds = 5,
  #sort_metric = "AUC",
  seed = 20220412,
  max_runtime_secs = 600, # limit the run-time
  project_name = "automl"
)
automl


automl_pred <- h2o.predict(automl, validation_h2o)
automl_pred <- as.data.frame(automl_pred)

# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = automl_pred[,1]) %>% 
  write.csv("../AS3/models/automl.csv", row.names = F) 


# DL ----
# default DL
dl_default <- h2o.deeplearning(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  validation_frame = test_h2o,
  model_id = "dl_default",
  seed = my_seed)

dl_default

h2o.scoreHistory(dl_default)
plot(dl_default, metric = "deviance")

dl_default_pred <- h2o.predict(dl_default, validation_h2o)
dl_default_pred <- as.data.frame(dl_default_pred)

# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = dl_default_pred[,1]) %>% 
  write.csv("../AS3/kaggle_submission_dl_default.csv", row.names = F) 

# DL adjusted 1

dl_adjusted_1 <- h2o.deeplearning(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  validation_frame = test_h2o,
  model_id = "dl_adjusted_1",
  hidden = c(400, 400),
  mini_batch_size = 20,
  score_each_iteration = TRUE,
  seed = my_seed
)

dl_adjusted_1

h2o.scoreHistory(dl_adjusted_1)
plot(dl_adjusted_1, metric = "deviance")

dl_adjusted_1_pred <- h2o.predict(dl_adjusted_1, test_h2o)
dl_adjusted_1_pred <- as.data.frame(dl_adjusted_1_pred)

# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = dl_adjusted_1_pred[,1]) %>% 
  write.csv("../AS3/kaggle_submission_dl_adjusted_1.csv", row.names = F) 

# DL adjusted 2

dl_regularized <- h2o.deeplearning(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  validation_frame = test_h2o,
  model_id = "dl_regularized",
  hidden = c(400, 400),
  mini_batch_size = 20,
  activation = "RectifierWithDropout",
  hidden_dropout_ratios = c(0.2, 0.2),
  epochs = 300,
  score_each_iteration = TRUE,
  seed = my_seed
)


dl_regularized

h2o.scoreHistory(dl_regularized)
plot(dl_regularized, metric = "deviance")

dl_regularized_pred <- h2o.predict(dl_regularized, validation_h2o)
dl_regularized_pred <- as.data.frame(dl_regularized_pred)

# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = dl_regularized_pred[,1]) %>% 
  write.csv("../AS3/kaggle_submission_dl_regularized.csv", row.names = F) 

# Deep net

dl_deep <- h2o.deeplearning(
  x = predictors,
  y = response,
  training_frame = train,
  validation_frame = test,
  model_id = "dl_deep",
  hidden = c(256, 128, 64, 32),
  mini_batch_size = 20,
  activation = "RectifierWithDropout",
  hidden_dropout_ratios = c(0.2, 0.2, 0.2, 0.2),
  epochs = 300,
  score_each_iteration = TRUE,
  seed = my_seed
)

dl_deep

h2o.scoreHistory(dl_deep)
plot(dl_deep, metric = "deviance")

dl_deep_pred <- h2o.predict(dl_deep, validation_h2o)
dl_deep_pred <- as.data.frame(dl_deep_pred)

# kaggle sumbission
select(test, article_id) %>% 
  mutate(score = dl_deep_pred[,1]) %>% 
  write.csv("../AS3/kaggle_submission_dl_deep.csv", row.names = F) 



