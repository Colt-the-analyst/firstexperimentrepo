################################################################################
# Modelling - Predicting EQI (Equity Index)
#
# The purpose of this script is to create a model that predicts a service's
# Equity Index (EQI) score based on other information available in the
# Early Childhood Education (ECE) directory.
#
# The model uses the multilayer perceptron (MLP) algorithm and applies
# hyperparameter tuning to optimize performance. It also includes feature
# importance analysis using permutation-based techniques.
#
# Created by:
#   Colton Brewer
#
# Date:
#   04/02/2025
################################################################################

# Load required libraries
library(tidyverse)     # For data manipulation and visualization
library(tidymodels)    # For machine learning pipeline
library(janitor)       # For data cleaning
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
directory_data <- directory_data |> filter(
  equity_index_eqi %in% c("EQI 1", "EQI 2", "EQI 3", "EQI 4", "EQI > 5"),!is.na(equity_index_eqi)
) |>
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

mlp_model <- 
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |> 
  set_engine("nnet", trace = 0) |> 
  set_mode("classification")

# Define a recipe for preprocessing
mlp_recipe <- recipe(equity_index_eqi ~ ., data = directory_train) |>
  step_normalize(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors())

mlp_wflow <-
  workflow() |>
  add_model(mlp_model) |>
  add_recipe(mlp_recipe)

mlp_param <- extract_parameter_set_dials(mlp_model)

# Generate a grid of 10 possible values for k (ranging from 1 to 50)
mlp_grid <- grid_regular(mlp_param, levels = 4)

mlp_grid

################################################################################
# Create Resampling Folds (Cross-Validation)
################################################################################

set.seed(111111)
directory_folds <- vfold_cv(directory_train, v = 5, strata = equity_index_eqi, repeats = 3)


# Enable parallel processing
plan(multisession, workers = 4)

# Perform grid search using cross-validation
rf_results <- tune_grid(
  mlp_wflow,
  resamples = directory_folds,
  grid = mlp_grid,
  metrics = metric_set(accuracy)
)

# Reset processing to sequential after tuning
plan(sequential)

mlp_fit <- fit(mlp_wflow, directory_train)
