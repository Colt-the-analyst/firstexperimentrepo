################################################################################
# Modelling - Predicting EQI (Equity Index)
#
# The purpose of this script is to create a model that predicts a service's
# Equity Index (EQI) score based on other information available in the
# Early Childhood Education (ECE) directory.
#
# The model uses the Random Forest algorithm and applies hyperparameter tuning
# to optimize performance. It also includes feature importance analysis using
# permutation-based techniques.
#
# Created by:
#   Colton Brewer
#
# Date:
#   02/02/2025
################################################################################

# Load required libraries
library(tidyverse)
library(tidymodels)
library(janitor)
library(ranger)        # Random Forest engine for tidymodels
library(future)        # Parallel processing

# Set tidymodels preference to avoid conflicts between packages
tidymodels_prefer()

################################################################################
# Step 1: Load and Prepare Data
################################################################################

# Define the data path
data_path <- file.path("Data", "directory.csv")

# Read the CSV file and clean column names
directory_data <- read.csv(data_path, skip = 16) |>
  select(-starts_with("X")) |>
  clean_names()

# Ensure `equity_index_eqi` has no missing values
directory_data <- directory_data |> 
  filter(!is.na(equity_index_eqi)) |>
  mutate(
    age_0_prop = age_0 / total,
    age_1_prop = age_1 / total,
    age_2_prop = age_2 / total,
    age_3_prop = age_3 / total,
    age_4_prop = age_4 / total,
    age_5_prop = age_5 / total,
    european_pakeha_prop = european_pakeha / total,
    maori_prop = maori / total,
    pacific_prop = pacific / total,
    asian_pakeha_prop = asian / total,
    other_race_prop = other / total,
    equity_index_eqi = as.factor(equity_index_eqi),
    community_of_learning_id = as.factor(case_when(
      !is.na(community_of_learning_id) ~ as.character(community_of_learning_id),
      is.na(community_of_learning_id) ~ "None"
    ))
  ) |> 
  select(-c(
    service_name, telephone, email, street, suburb, definition,
    territorial_authority, general_electorate, maori_electorate,
    neighbourhood_sa2_code, neighbourhood_sa2, ward,
    management_contact_name, management_postal_address,
    management_postal_suburb, management_postal_city,
    management_contact_phone, community_of_learning_name,
    age_0, age_1, age_2, age_3, age_4, age_5,
    european_pakeha, maori, pacific, asian, other
  )) |> 
  filter(!is.na(total))

# Split the data into training (80%) and testing (20%) sets
set.seed(11)
directory_split <- initial_split(directory_data, prop = 0.8, strata = equity_index_eqi)
directory_train <- training(directory_split)
directory_test <- testing(directory_split)

# 🛠 **Fix 1: Ensure same levels in training & testing**
directory_train <- directory_train |> mutate(equity_index_eqi = fct_drop(equity_index_eqi))
directory_test <- directory_test |> mutate(equity_index_eqi = factor(equity_index_eqi, levels = levels(directory_train$equity_index_eqi)))

################################################################################
# Step 2: Preprocessing - Create a Recipe
################################################################################

# Define a preprocessing recipe for Random Forest
rf_recipe <- recipe(equity_index_eqi ~ ., data = directory_train) |>
  step_indicate_na(longitude, latitude) |>
  step_novel(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.01) |>
  step_zv(all_predictors()) |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = FALSE)

# Prepare the recipe
rf_prep <- prep(rf_recipe, training = directory_train)

################################################################################
# Step 3: Create Resampling Folds (Cross-Validation)
################################################################################

set.seed(111111)
directory_folds <- vfold_cv(directory_train, v = 5, strata = equity_index_eqi, repeats = 3)

################################################################################
# Step 4: Define the Random Forest Model with Hyperparameter Tuning
################################################################################

# Define a tunable Random Forest model
rf_model <-
  rand_forest(mtry = tune(), trees = tune(), min_n = tune()) |>
  set_engine("ranger") |>
  set_mode("classification")

################################################################################
# Step 5: Finalize `mtry` and Create a Grid Search for Hyperparameters
################################################################################

# Extract tunable parameters
rf_param <- extract_parameter_set_dials(rf_model)

# 🛠 **Fix 2: Properly finalize `mtry`**
rf_param <- finalize(rf_param, directory_train)

# 🛠 **Fix 3: Use `grid_random()` instead of `grid_regular()`**
rf_grid <- grid_random(rf_param, size = 10)

################################################################################
# Step 6: Create a Workflow for Tuning
################################################################################

# Create a workflow that bundles the model with preprocessing
rf_wflow_tune <- workflow() |>
  add_model(rf_model) |>
  add_recipe(rf_recipe)

################################################################################
# Step 7: Tune the Model Using `tune_grid()` with Parallel Processing
################################################################################

set.seed(111111)

# Enable parallel processing
plan(multisession, workers = 4)

# Perform grid search using cross-validation
rf_results <- tune_grid(
  rf_wflow_tune,
  resamples = directory_folds,
  grid = rf_grid,
  metrics = metric_set(accuracy)
)

# Reset processing to sequential after tuning
plan(sequential)

################################################################################
# Step 8: Debugging - Run on a Single Fold if Errors Occur
################################################################################

# 🛠 **Fix 4: Run `tune_grid()` on a single fold for debugging**
rf_results_single <- tune_grid(
  rf_wflow_tune,
  resamples = slice(directory_folds, 1),
  grid = rf_grid,
  metrics = metric_set(accuracy)
)

################################################################################
# Step 9: Visualize the Tuning Results
################################################################################

autoplot(rf_results)

################################################################################
# Step 10: Finalize the Model with the Best Hyperparameters
################################################################################

best_rf_params <- rf_results |> select_best(metric = "accuracy")
final_rf_model <- finalize_model(rf_model, best_rf_params)

final_rf_wflow <- workflow() |>
  add_model(final_rf_model) |>
  add_recipe(rf_recipe)

# Train the final model
final_rf_fit <- fit(final_rf_wflow, data = directory_train)

################################################################################
# Step 11: Make Predictions on the Test Set
################################################################################

rf_test_res <- predict(final_rf_fit, new_data = directory_test |> select(-equity_index_eqi)) |>
  bind_cols(directory_test |> select(equity_index_eqi))

################################################################################
# Step 12: Evaluate Final Model Performance
################################################################################

accuracy(rf_test_res, truth = equity_index_eqi, estimate = .pred_class)
conf_mat(rf_test_res, truth = equity_index_eqi, estimate = .pred_class)

################################################################################
# Step 13: Compute Additional Performance Metrics
################################################################################

metrics <- metric_set(accuracy, precision, recall, f_meas)
metrics(rf_test_res, truth = equity_index_eqi, estimate = .pred_class, na_rm = TRUE)

################################################################################
# End of Script
################################################################################