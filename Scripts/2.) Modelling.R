################################################################################
# Modelling
# 
# The purpose of this file is to create a model that predicts a services EQI
# based on other information that is available from the ECE directory
# 
# Created by:
#   Colton Brewer
# 
# Date:
#   30/01/2025
# 
################################################################################

# Load required libraries
library(tidyverse)
library(tidymodels)
library(janitor)
library(kknn)
library(vip)  # For feature importance

tidymodels_prefer()

# Step 1: Ensure factor levels are consistent in training and test data
directory_train <- directory_train %>%
  mutate(equity_index_eqi = factor(equity_index_eqi))

directory_test <- directory_test %>%
  mutate(equity_index_eqi = factor(equity_index_eqi, levels = levels(directory_train$equity_index_eqi)))

# Step 2: Create resampling folds (Cross-validation)
set.seed(111111)
directory_folds <- vfold_cv(directory_train, v = 10, strata = equity_index_eqi)  # 10-fold CV for better generalization

# Step 3: Modify KNN model to tune `neighbors` with weighted distance
knn_model <-
  nearest_neighbor(neighbors = tune(), weight_func = "inv") |>  # Use inverse weighting for better class handling
  set_engine("kknn") |>
  set_mode("classification")

# Step 4: Create a new workflow for tuning
knn_wflow_tune <- 
  workflow() |>
  add_model(knn_model) |>
  add_formula(equity_index_eqi ~ .)

# Step 5: Define the grid search for `neighbors`
knn_grid <- grid_regular(neighbors(range = c(1, 100)), levels = 20)  # Test 20 values between k = 1 and 100

# Step 6: Tune the model using `tune_grid()`
set.seed(111111)
knn_results <- tune_grid(
  knn_wflow_tune,
  resamples = directory_folds,  # Use cross-validation
  grid = knn_grid,
  metrics = metric_set(accuracy)
)

# Step 7: Visualize tuning results
autoplot(knn_results)

# Step 8: Select the best k-value based on accuracy
best_k <- knn_results |> select_best(metric = "accuracy")

# Step 9: Finalize the model with the best k-value
final_knn_model <- finalize_model(knn_model, best_k)

# Step 10: Create final workflow and fit it on the full training data
final_knn_wflow <- 
  workflow() |>
  add_model(final_knn_model) |>
  add_formula(equity_index_eqi ~ .)

final_knn_fit <- fit(final_knn_wflow, data = directory_train)

# Step 11: Visualize Feature Importance using `vip`
# Define a prediction function
predict_class_knn <- function(model, newdata) {
  predict(model, newdata)$.pred_class  # Extract class predictions
}

# Compute permutation-based feature importance
vi_knn <- vi_permute(
  final_knn_fit,
  train = directory_train,
  target = "equity_index_eqi",
  metric = "accuracy",
  pred_wrapper = predict_class_knn  # Use custom prediction function
)

# Visualize feature importance
vip(vi_knn)

# Step 12: Make predictions on the test set
knn_test_res <- predict(final_knn_fit, new_data = directory_test |> select(-equity_index_eqi)) |>
  bind_cols(directory_test |> select(equity_index_eqi))

# Step 13: Evaluate Final Model Performance

# Compute Accuracy
accuracy(knn_test_res, truth = equity_index_eqi, estimate = .pred_class)

# Compute Confusion Matrix
conf_mat(knn_test_res, truth = equity_index_eqi, estimate = .pred_class)

# Step 14: Compute Precision, Recall, and F1 Score with NA handling
metrics <- metric_set(
  accuracy,
  precision,
  recall,
  f_meas
)

metrics(knn_test_res, truth = equity_index_eqi, estimate = .pred_class, na_rm = TRUE)
