################################################################################
# Modelling - Predicting EQI (Equity Index)
# 
# The purpose of this script is to create a model that predicts a service's 
# Equity Index (EQI) score based on other information available in the 
# Early Childhood Education (ECE) directory.
# 
# The model uses the k-Nearest Neighbors (KNN) algorithm and applies 
# hyperparameter tuning to optimize performance. It also includes feature 
# importance analysis using permutation-based techniques.
# 
# Created by:
#   Colton Brewer
# 
# Date:
#   30/01/2025
################################################################################

# Load required libraries
library(tidyverse)     # For data manipulation and visualization
library(tidymodels)    # For machine learning pipeline
library(janitor)       # For data cleaning
library(kknn)          # KNN model implementation
library(vip)           # Variable importance for models

# Set tidymodels preference to avoid conflicts between packages
tidymodels_prefer()

################################################################################
# Step 1: Ensure Factor Levels are Consistent in Training and Test Data
################################################################################

# Convert `equity_index_eqi` to a factor in the training set
directory_train <- directory_train %>%
  mutate(equity_index_eqi = factor(equity_index_eqi))

# Ensure test data uses the same factor levels as the training set
directory_test <- directory_test %>%
  mutate(equity_index_eqi = factor(equity_index_eqi, levels = levels(directory_train$equity_index_eqi)))

# Why?
# - Machine learning models in R treat categorical variables (`factor` type) 
#   differently from numerical variables.
# - Ensuring test set uses the same levels as training prevents errors during 
#   prediction.

################################################################################
# Step 2: Create Resampling Folds (Cross-Validation)
################################################################################

set.seed(111111)  # Ensure reproducibility

# Create a 10-fold cross-validation object
directory_folds <- vfold_cv(directory_train, v = 10, strata = equity_index_eqi)

# Why?
# - Cross-validation (CV) helps estimate model performance on unseen data.
# - Stratification (`strata = equity_index_eqi`) ensures class balance 
#   across folds.

################################################################################
# Step 3: Define the KNN Model with Hyperparameter Tuning
################################################################################

# Define a k-Nearest Neighbors model with a tunable `neighbors` parameter
knn_model <-
  nearest_neighbor(neighbors = tune(), weight_func = "inv") |>  
  set_engine("kknn") |>
  set_mode("classification")

# Why?
# - `neighbors = tune()` tells `tidymodels` to search for the best k-value.
# - `weight_func = "inv"` assigns **higher weights to closer neighbors**, 
#   improving classification for imbalanced datasets.

################################################################################
# Step 4: Create a Workflow for Tuning
################################################################################

# Create a `tidymodels` workflow that bundles the model with the formula
knn_wflow_tune <- 
  workflow() |>
  add_model(knn_model) |>
  add_formula(equity_index_eqi ~ .)  # Use all features to predict `equity_index_eqi`

################################################################################
# Step 5: Define the Grid Search for `neighbors`
################################################################################

# Generate a grid of 20 possible values for k (ranging from 1 to 100)
knn_grid <- grid_regular(neighbors(range = c(1, 100)), levels = 20)

################################################################################
# Step 6: Tune the Model Using `tune_grid()`
################################################################################

set.seed(111111)  # Ensure reproducibility

# Perform grid search using cross-validation
knn_results <- tune_grid(
  knn_wflow_tune,
  resamples = directory_folds,  # Use cross-validation folds
  grid = knn_grid,              # Use predefined k values
  metrics = metric_set(accuracy) # Evaluate models using accuracy
)

################################################################################
# Step 7: Visualize the Tuning Results
################################################################################

# Plot how accuracy changes across different k-values
autoplot(knn_results)

################################################################################
# Step 8: Select the Best k-Value Based on Accuracy
################################################################################

# Identify the best k-value that maximizes accuracy
best_k <- knn_results |> select_best(metric = "accuracy")

################################################################################
# Step 9: Finalize the Model with the Best k-Value
################################################################################

# Finalize the KNN model using the best k-value from tuning
final_knn_model <- finalize_model(knn_model, best_k)

################################################################################
# Step 10: Train the Final Model on the Full Training Data
################################################################################

final_knn_wflow <- 
  workflow() |>
  add_model(final_knn_model) |>
  add_formula(equity_index_eqi ~ .)

# Train the final model
final_knn_fit <- fit(final_knn_wflow, data = directory_train)

################################################################################
# Step 11: Analyze Feature Importance Using `vip::vi_permute()`
################################################################################

# Define a prediction function to extract class predictions
predict_class_knn <- function(model, newdata) {
  predict(model, newdata)$.pred_class  
}

# Compute permutation-based feature importance using a sample for efficiency
set.seed(123)
subset_train <- directory_train %>% sample_n(300)  # Speed up computation

vi_knn <- vi_permute(
  final_knn_fit,
  train = subset_train,
  target = "equity_index_eqi",
  metric = "accuracy",
  pred_wrapper = predict_class_knn,
  nsim = 3  # Reduce number of permutations to speed up processing
)

# Plot feature importance
vip(vi_knn)

# Why?
# - `vi_permute()` shuffles features and measures how accuracy changes.
# - Helps understand **which features impact predictions the most**.

################################################################################
# Step 12: Make Predictions on the Test Set
################################################################################

# Generate predictions and bind the actual EQI values
knn_test_res <- predict(final_knn_fit, new_data = directory_test |> select(-equity_index_eqi)) |>
  bind_cols(directory_test |> select(equity_index_eqi))

################################################################################
# Step 13: Evaluate Final Model Performance
################################################################################

# Compute accuracy of the final model on the test set
accuracy(knn_test_res, truth = equity_index_eqi, estimate = .pred_class)

# Compute the confusion matrix to visualize misclassifications
conf_mat(knn_test_res, truth = equity_index_eqi, estimate = .pred_class)

################################################################################
# Step 14: Compute Additional Performance Metrics
################################################################################

# Define a metric set to evaluate classification performance
metrics <- metric_set(
  accuracy,  # Overall correctness
  precision, # How many predicted positives were correct?
  recall,    # How many actual positives were predicted correctly?
  f_meas     # Harmonic mean of precision and recall
)

# Compute the metrics, ignoring missing classes (to prevent warnings)
metrics(knn_test_res, truth = equity_index_eqi, estimate = .pred_class, na_rm = TRUE)

################################################################################
# End of Script
################################################################################
