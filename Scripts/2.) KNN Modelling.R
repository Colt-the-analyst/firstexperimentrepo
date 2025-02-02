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
#   01/02/2025
################################################################################

# Load required libraries
library(tidyverse)     # For data manipulation and visualization
library(tidymodels)    # For machine learning pipeline
library(janitor)       # For data cleaning
library(kknn)          # KNN model implementation
library(vip)           # Variable importance for models
library(future)        # For parallel processing

# Set tidymodels preference to avoid conflicts between packages
tidymodels_prefer()

################################################################################
# Step 1: Load and Prepare Data
################################################################################

# Define the data path
data_path <- file.path("Data", "directory.csv")

# Read the CSV file and clean column names
directory_data <- read.csv(data_path, skip = 16) |> 
  select(-starts_with("X")) |>  # Remove unnecessary columns
  clean_names()

# Ensure `equity_index_eqi` has no missing values
directory_data <- directory_data |> filter(!is.na(equity_index_eqi)) |>
  mutate(
    age_0_prop = age_0/total,
    age_1_prop = age_1/total,
    age_2_prop = age_2/total,
    age_3_prop = age_3/total,
    age_4_prop = age_4/total,
    age_5_prop = age_5/total,
    european_pakeha_prop = european_pakeha/total,
    maori_prop = maori/total,
    pacific_prop = pacific/total,
    asian_pakeha_prop = asian/total,
    other_race_prop = other/total,
    equity_index_eqi = as.factor(equity_index_eqi),
    community_of_learning_id = as.factor(
      case_when(
        !is.na(community_of_learning_id) ~ as.character(community_of_learning_id),
        is.na(community_of_learning_id) ~ "None"
      ))
  ) |>
  select(
    -c(
      service_name,
      telephone,
      email,
      street,
      suburb,
      definition,
      territorial_authority,
      general_electorate,
      maori_electorate,
      neighbourhood_sa2_code,
      neighbourhood_sa2,
      ward,
      management_contact_name,
      management_postal_address,
      management_postal_suburb,
      management_postal_city,
      management_contact_phone,
      community_of_learning_name,
      age_0,
      age_1,
      age_2,
      age_3,
      age_4,
      age_5,
      european_pakeha,
      maori,
      pacific,
      asian,
      other
    )
  ) |>
  filter(!is.na(total))

# Split the data into training (80%) and testing (20%) sets, ensuring stratification
set.seed(11)
directory_split <- initial_split(directory_data, prop = 0.8, strata = equity_index_eqi)
directory_train <- training(directory_split)
directory_test <- testing(directory_split)

dim(directory_split)
dim(directory_train)
dim(directory_test)

################################################################################
# Step 2: Preprocessing - Create a Recipe
################################################################################

# Define a preprocessing recipe
knn_recipe <- recipe(equity_index_eqi ~ ., data = directory_train) |>
  step_indicate_na(longitude, latitude)  |>
  step_novel(all_nominal_predictors()) |>  # Handle unseen factor levels
  step_other(all_nominal_predictors(), threshold = 0.01) |>  # Group rare levels
  step_dummy(all_nominal_predictors()) |>  # Convert categorical variables to dummy variables
  step_zv(all_numeric_predictors()) |>  # Remove zero-variance predictors
  step_normalize(all_numeric_predictors())  # Normalize numerical predictors 
  
knn_recipe

# Prepare the recipe
knn_prep <- prep(knn_recipe, training = directory_train)

knn_prep

################################################################################
# Step 3: Create Resampling Folds (Cross-Validation)
################################################################################

set.seed(111111)  # Ensure reproducibility

# Create a 5-fold cross-validation object for efficiency
directory_folds <- vfold_cv(directory_train, v = 5, strata = equity_index_eqi, repeats = 10)

directory_folds

################################################################################
# Step 4: Define the KNN Model with Hyperparameter Tuning
################################################################################

# Define a k-Nearest Neighbors model with a tunable `neighbors` parameter
knn_model <-
  nearest_neighbor(neighbors = tune(), weight_func = "rectangular") |>  
  set_engine("kknn") |>
  set_mode("classification")

knn_model

################################################################################
# Step 5: Create a Workflow for Tuning
################################################################################

# Create a workflow that bundles the model with preprocessing
knn_wflow_tune <- 
  workflow() |>
  add_model(knn_model) |>
  add_recipe(knn_recipe)

knn_wflow_tune

################################################################################
# Step 6: Define a Grid Search for `neighbors`
################################################################################

# Generate a grid of 10 possible values for k (ranging from 1 to 50)
knn_grid <- grid_regular(neighbors(range = c(1, 20)), levels = 20)

knn_grid

################################################################################
# Step 7: Tune the Model Using `tune_grid()` with Parallel Processing
################################################################################

set.seed(111111)  # Ensure reproducibility

# Enable parallel processing to speed up tuning (limit to 4 CPU cores)
plan(multisession, workers = 4)

# Perform grid search using cross-validation
knn_results <- tune_grid(
  knn_wflow_tune,
  resamples = directory_folds,
  grid = knn_grid,
  metrics = metric_set(accuracy)
)

# Reset processing to sequential after tuning
plan(sequential)

################################################################################
# Step 8: Visualize the Tuning Results
################################################################################

autoplot(knn_results)

################################################################################
# Step 9: Finalize the Model
################################################################################

best_k <- knn_results |> select_best(metric = "accuracy")
final_knn_model <- finalize_model(knn_model, best_k)

final_knn_wflow <- 
  workflow() |>
  add_model(final_knn_model) |>
  add_recipe(knn_recipe)

final_knn_fit <- fit(final_knn_wflow, data = directory_train)

################################################################################
# Step 12: Analyze Feature Importance Using `vip::vi_permute()`
################################################################################

# Define a prediction function to extract class predictions
predict_class_knn <- function(model, newdata) {
  predict(model, newdata)$.pred_class  
}

# Compute permutation-based feature importance using a smaller sample for efficiency
set.seed(123)
subset_train <- directory_train |> sample_n(300)  # Speed up computation

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

################################################################################
# Step 13: Make Predictions on the Test Set
################################################################################

# Generate predictions and bind the actual EQI values
knn_test_res <- predict(final_knn_fit, new_data = directory_test |> select(-equity_index_eqi)) |>
  bind_cols(directory_test |> select(equity_index_eqi))

################################################################################
# Step 14: Evaluate Final Model Performance
################################################################################

# Compute accuracy of the final model on the test set
accuracy(knn_test_res, truth = equity_index_eqi, estimate = .pred_class)

# Compute the confusion matrix to visualize misclassifications
conf_mat(knn_test_res, truth = equity_index_eqi, estimate = .pred_class)

################################################################################
# Step 15: Compute Additional Performance Metrics
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