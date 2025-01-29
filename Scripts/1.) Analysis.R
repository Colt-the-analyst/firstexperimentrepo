################################################################################
# Wrangling
# 
# The purpose of this file is to perform exploratory data analysis (EDA) on the
# early childhood education (ECE) data.
# 
# Created by:
#   Colton Brewer
# 
# Date:
#   29/01/2025
# 
################################################################################

# Step 1
# Load the packages
library(tidyverse)
library(skimr)
library(DataExplorer)
library(janitor)
library(ggcorrplot)

# Step 2
# Load the raw data
data_path <- file.path("Data", "directory.csv")
# Read the csv file
directory_data <- read.csv(data_path, skip = 16) |>
  select(-starts_with("X")) |>
  clean_names()

# Step 3
# Check the Structure of the Data
glimpse(directory_data)

dim(directory_data)

# Step 4
# Wrangle the data to an appropriate format
directory_data <- directory_data |>
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
    community_of_learning_id = as.factor(community_of_learning_id)
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
)

# Step 5
# Check for Missing Values
sum(is.na(directory_data))   # Total number of missing values
colSums(is.na(directory_data))  # Missing values per column

plot_missing(directory_data)  # From DataExplorer package

# Step 6
# Summary Statistics
summary(directory_data)  # Basic statistics for numeric columns

skim(directory_data)

# Step 7
# Identify Duplicate Rows
sum(duplicated(directory_data))
# [1] 0
# All rows represent different services so this makes sense.

# Step 8
# Check Unique Values in Categorical Variables
directory_data |>
  summarise(across(where(is.character), n_distinct))

# To preview the first few unique values:
directory_data |>
  select(where(is.character)) |>
  map(unique)

# Step 9
# Univariate Analysis (Single Variables)
directory_data |> select(where(is.numeric)) |> summary()

directory_data |> 
  select(where(is.numeric)) |> 
  gather() |> 
  ggplot(aes(value)) + 
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) + 
  facet_wrap(~key, scales = "free") +
  theme_minimal()

# Step 10
# Bivariate Analysis (Relationships Between Variables)
correlation_matrix <- cor(directory_data %>%
                            select(where(is.numeric)), use = "complete.obs")
print(correlation_matrix)

ggcorrplot(correlation_matrix, lab = TRUE)

# Step 11
# Automated EDA Report
create_report(directory_data)

