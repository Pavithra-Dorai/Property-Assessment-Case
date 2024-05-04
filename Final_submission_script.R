
## Load necessary libraries 

library(tidyverse)
library(glmnet)

## Load and Clean Historical Property Data

data <- read.csv('historic_property_data.csv')
str(data)
head(data)

## Exclude columns that are not predictors

# Specify the column indices to be removed
columns_to_remove <- c(2, 5, 6, 7, 8, 27, 28, 33, 35, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61)

# Remove the specified columns
data <- subset(data, select = -columns_to_remove)

# Print the structure of the updated data
str(data)
## Rename Columns

names(data) <- c("sale_price","Town_Code","Neighborhood_Code","Land_sqft","Age_Apt","Num_Apt","Wall_Material","Roof_Material","Room_Num","Bedroom_Num","Basement","Basement_Fin","Heat_Central","Heat_Other","AC_Central","Fireplace","Attic_Type","Bath_Full","Bath_Half","Design_Plan","Ceiling_Cathedral","Garage_Size","Garage_Material","Garage_Attached","Garage_Area","Building_sqft","Usage_Type","Residence_Type","Attic_Fin","Porch","Noise_Ind","Floodplain_FEMA","Risk_Factor_Flood","Risk_Direction_Flood","Proximity_Road_100","Proximity_Road_101_300","Dist_Elem_School","Dist_High_School","Tax_rate","Income_Median","Indicator_Garage","armslength_ind")
str(data)

### Handle columns with high percentage of null values
# Determine the number of nulls for each column

# Function that counts the amount of nulls per column
null_count_column <- function(dataset) {
  
  # Use colSums and is.na to count null values in each column
  null_counts <- colSums(is.na(dataset))
  
  # Create a data frame with column names and corresponding null counts
  result_table <- data.frame(
    Column = names(null_counts),
    Null_Count = null_counts
  )
  
  return(result_table)
}

# This shows us the NAs count for each column
result <- null_count_column(data)
print(arrange(result, desc(Null_Count)))

## Set a threshold for NAs and remove columns that comply with the condition

# Set threshold of 10% to drop columns that have over 10% of NAs
threshold <- 0.1
columns_to_remove <- names(data)[colSums(is.na(data)) / nrow(data) > threshold]

# Remove columns with more than 10% NAs
data <- data[, -which(names(data) %in% columns_to_remove)]
str(data)

### Check the columns that still have NAs

sapply(data, function(x) sum(is.na(x)))

## Remove observations with null values

data <- na.omit(data)



# Define a function to replace outliers with the threshold values using Z-Score
replace_outliers_with_threshold <- function(x, threshold = 3) {
  z_scores <- abs(scale(x))
  x[z_scores > threshold] <- threshold
  x
}

# Apply the function to multiple columns
numeric_vars <- sapply(data, is.numeric)
data[numeric_vars] <- lapply(data[numeric_vars], replace_outliers_with_threshold)

# Filter out rows that will skew our model as well as columns that have too low variation

data <- data %>% filter(sale_price < 1000000) %>% select(-Risk_Direction_Flood, -Floodplain_FEMA, -Noise_Ind, -Proximity_Road_100, -Proximity_Road_101_300)

##### Backward Elimination #####


# Load the package
library(leaps)

# Partion the data
# set seed for reproducing the partition 
set.seed(1)  

# row numbers of the training set 
train.index <- sample(c(1:dim(data)[1]), 0.5*dim(data)[1]) 

# training set 
train.df <- data[train.index, ]
dim(train.df)

# test set 
test.df <- df[-train.index, ]
dim(test.df)

# fit the model with all predictors  
lm.full <- lm(sale_price ~ ., data = train.df)

# use step() to run backward elimination 
lm.step.backward <- step(lm.full, direction = "backward")

# summary table 
summary(lm.step.backward)

# making predictions on the test set
lm.step.pred.backward <- predict(lm.step.backward , test.df)

# MSE in the test set 
mean((test.df$sale_price - lm.step.pred.backward) ^ 2)


##### Forward Selection #####

# create model with no predictors
lm.null <- lm(sale_price ~ 1, data = train.df)
lm.null

# use step() to run forward selection  
lm.step.forward <- step(lm.null, scope=list(lower=lm.null, upper=lm.full), direction = "forward")

# summary table 
summary(lm.step.forward)

# making predictions on the test set
lm.step.pred.forward <- predict(lm.step.forward, test.df)

# MSE in the test set 
mean((test.df$sale_price - lm.step.pred.forward) ^ 2)

##### stepwise regression #####

# use step() to run stepwise regression  
lm.step.both <- step(lm.full, direction = "both")

# summary table 
summary(lm.step.both)

# make predictions on the test set
lm.step.pred.both <- predict(lm.step.both, test.df)

# MSE in the test set 
mean((test.df$sale_price - lm.step.pred.both) ^ 2)



##### Regression Trees #####

# Load the packages
library(rpart)
library(rpart.plot)

# regression tree with cp = 0.01
rt.deep <- rpart(sale_price ~ ., data = train.df, cp = 0.01, method = "anova", minbucket = 1, maxdepth = 30)

# plot the tree
prp(rt.deep, type = 1, extra = 1)

# predicted prices for records in the test set 
rt.deep.pred <- predict(rt.deep, test.df, type="vector")

# MSE in the test set 
mean((test.df$sale_price - rt.deep.pred) ^ 2)


# prune the regression tree
# set the seed 
set.seed(1)

# fit a regression tree with cp = 0.0001 and xval = 5
cv.rt <- rpart(sale_price ~ ., data = train.df, cp = 0.001, xval = 10, method = "anova")

# display the cp table
cv.rt$cptable

# prune the tree
rt.pruned <- prune(cv.rt, cp = cv.rt$cptable[which.min(cv.rt$cptable[,"xerror"]),"CP"])

# plot the tree
prp(rt.pruned, type = 1, extra = 1)

# predicted prices for records in the test set 
rt.pruned.pred <- predict(rt.pruned, test.df, type = "vector")

# MSE in the test set 
mean((test.df$sale_price - rt.pruned.pred) ^ 2)




##### Lasso Regression. #####

## Convert data into a matrix

# Convert a data frame of predictors to a matrix and create dummy variables for character variables 
x <- model.matrix(sale_price~.,data)[,-1]

# Outcome 
y <- data$sale_price

## Partition Data

# Set seed 
set.seed(1)

# Row numbers of the training set 
train_rows <- sample(c(1:dim(x)[1]), dim(x)[1]*0.6)
length(train_rows)


# Row numbers of the test set 
test_rows <- setdiff(c(1:dim(data)[1]),train_rows)
length(test_rows)
rm(data)

# Outcome in the test set 
y.test <- y[test_rows]


# Fit a lasso regression model 
lasso_fit <- glmnet(x[train_rows,],y[train_rows],alpha=1)

# Sequence of lambda values 
lasso_fit$lambda

# Dimension of lasso regression coefficients 
dim(coef(lasso_fit))

# Plot coefficients on log of lambda values 
plot(lasso_fit, xvar="lambda")

## Using Cross Validation to choose the appropriate Lambda value

# Set seed 
set.seed(1)

# 10-fold cross validation 
cv_lasso_fit <- cv.glmnet(x[train_rows,],y[train_rows],alpha=1, type.measure="mse", nfold=10)

# Plot the cross-validated MSE for each lambda 
plot(cv_lasso_fit)

# Lambda that corresponds to the lowest cross-validated MSE 
best_lambda <- cv_lasso_fit$lambda.min
best_lambda

## Model with the Best Lambda

# Lasso regression coefficients  
coef_best_lambda <- predict(cv_lasso_fit,s=best_lambda,type="coefficients")
coef_best_lambda

# Non-zero coefficients 
coef_best_lambda[coef_best_lambda!=0] 

# Make predictions for records in the test set 
pred_best_lambda <- predict(lasso_fit,s=best_lambda,newx=x[test_rows,])
head(pred_best_lambda)

# MSE in the test set
MSE <- mean((y.test-pred_best_lambda)^2)
MSE



##### Predict_property_data#####

predict_df <- read.csv('predict_property_data.csv')
str(predict_df)
head(predict_df)

### Removing columns that are not predictors

# Specify the column indices to be removed
columns_to_remove <- c(2, 5, 6, 7, 8, 27, 28, 33, 35, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61)

# Subset the data to keep only the specified columns
predict_df <- subset(predict_df, select = -columns_to_remove)

# Print the structure of the updated data
str(predict_df)


### Renaming Columns

names(predict_df) <- c("pid","Town_Code","Neighborhood_Code","Land_sqft","Age_Apt","Num_Apt","Wall_Material","Roof_Material","Room_Num","Bedroom_Num","Basement","Basement_Fin","Heat_Central","Heat_Other","AC_Central","Fireplace","Attic_Type","Bath_Full","Bath_Half","Design_Plan","Ceiling_Cathedral","Garage_Size","Garage_Material","Garage_Attached","Garage_Area","Building_sqft","Usage_Type","Residence_Type","Attic_Fin","Porch","Noise_Ind","Floodplain_FEMA","Risk_Factor_Flood","Risk_Direction_Flood","Proximity_Road_100","Proximity_Road_101_300","Dist_Elem_School","Dist_High_School","Tax_rate","Income_Median","Indicator_Garage","armslength_ind")
str(predict_df)

### Checking the columns that still have NAs
sapply(predict_df, function(x) sum(is.na(x)))

### Checking the columns that still have NAs
# Install and load the mice package if not already installed
if (!requireNamespace("mice", quietly = TRUE)) {
  install.packages("mice")
}
library(mice)

# Impute missing values using mice with "pmm" method
mice_obj <- suppressMessages(mice(predict_df, method = "pmm"))

# Complete the imputation
predict_df_imputed <- suppressMessages(complete(mice_obj))

# Check for remaining missing values
sapply(predict_df_imputed, function(x) sum(is.na(x)))

str(predict_df_imputed)


## Change to categorical variables

columns_to_factor <- c("Wall_Material","Roof_Material","Basement","Heat_Central","Garage_Size","Residence_Type","Dist_Elem_School", "Dist_High_School", "Town_Code", "Neighborhood_Code", "Basement_Fin", "Heat_Other", "Heat_Central", "AC_Central", "Attic_Type", "Usage_Type", "Noise_Ind", "Floodplain_FEMA", "Proximity_Road_100", "Proximity_Road_101_300", "Indicator_Garage", "armslength_ind", "Garage_Material", "Garage_Area", "Garage_Attached")
for (col in columns_to_factor) {
  predict_df_imputed[[col]] <- as.factor(predict_df_imputed[[col]])
}
str(predict_df_imputed)


# select function to remove unnecessary columns
predict_df_imputed <- predict_df_imputed %>% select(-Num_Apt, -Porch, -Attic_Fin, -Ceiling_Cathedral, -Design_Plan, -Risk_Direction_Flood, -Floodplain_FEMA, -Noise_Ind, -Proximity_Road_100, -Proximity_Road_101_300)

# Check the structure of the data frame
str(predict_df_imputed)

# Convert predicted property data into a matrix
predict_df_imputed <- model.matrix(pid~., predict_df_imputed)[,-1]
dim(predict_df_imputed)

# Assuming colnames(x) and colnames(predict_df_imputed) represent the column names of x and predict_df_imputed
all_cols <- colnames(x)

# Identify missing columns
missing_cols <- setdiff(all_cols, colnames(predict_df_imputed))

# Add missing columns to df with NA values
predict_df_imputed <- cbind(predict_df_imputed, matrix(0, nrow = nrow(predict_df_imputed), ncol = length(missing_cols), dimnames = list(NULL, missing_cols)))

# Reorder columns to match the order in x
predict_df_imputed <- predict_df_imputed[, all_cols]
dim(predict_df_imputed)


# Convert predict_df_imputed into a matrix
predict_df_imputed <- as.matrix(predict_df_imputed)

# Predict housing prices from prediction dataset
preds <- predict(lasso_fit, s=best_lambda, newx = predict_df_imputed)
head(preds)

# Extract pid from the new data
pid <- predict_df$pid

# Create a data frame with pid and predicted assessed_value
result_df <- data.frame(pid = pid, assessed_value = preds)


# View critical values of distribution
summary(result_df$assessed_value)

# Write the result to a CSV file
write.csv(result_df, file = "assessed_value.csv", row.names = FALSE)
