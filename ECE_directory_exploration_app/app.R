# ---------------------------------------------------
# Load Required Libraries
# ---------------------------------------------------
library(shiny)      # For creating the interactive web app
library(dplyr)      # For data manipulation
library(ggplot2)    # For visualizing the data
library(DT)         # For interactive tables
library(janitor)    # For cleaning column names
library(readr)      # For reading CSV files

# ---------------------------------------------------
# Set the Working Directory (Modify this for your system)
# ---------------------------------------------------
setwd("~/R files/Happy Git and GitHub for the useR/firstexperimentrepo")

# ---------------------------------------------------
# Load and Clean Dataset
# ---------------------------------------------------

# Define the file path to the dataset
data_path <- file.path("Data", "directory.csv")

# Read the dataset, skipping the first 16 lines (to remove metadata rows)
directory_data <- read_csv(data_path, skip = 16, show_col_types = FALSE) %>%
  
  # Remove empty columns
  select(where(~ sum(!is.na(.)) > 0)) %>%
  
  # Clean column names (convert to lowercase and remove spaces)
  clean_names()

# Convert "equity_index_eqi" column to a categorical variable (factor)
directory_data$equity_index_eqi <- as.factor(directory_data$equity_index_eqi)

# ---------------------------------------------------
# Define UI (User Interface)
# ---------------------------------------------------
ui <- fluidPage(
  
  # Title of the App
  titlePanel("ECE Directory Data Explorer"),
  
  # Sidebar Layout (User Controls)
  sidebarLayout(
    sidebarPanel(
      
      # Dropdown: Select Regional Variable for the X-Axis
      selectInput("regionVar", "Select Regional Variable (X-axis):", 
                  choices = c("regional_council", "territorial_authority", "urban_rural", "ward", "education_region"),
                  selected = "regional_council"),
      
      # Dropdown: Select Variable for Stacking (Grouping)
      selectInput("groupVar", "Select Stacking Variable:", 
                  choices = c("regional_council", "territorial_authority", "urban_rural", "ward", "education_region"),
                  selected = "urban_rural"),
      
      # Slider: Adjust the Number of Bins in Histograms
      sliderInput("bins", "Number of Bins:", min = 5, max = 50, value = 20)
    ),
    
    # Main Panel (Displays Charts)
    mainPanel(
      tabsetPanel(
        tabPanel("Histogram of Services", plotOutput("histogramPlot")),
        tabPanel("Total Roll by Race", plotOutput("racePlot")),
        tabPanel("Total Roll by EQI", plotOutput("eqiPlot")),
        tabPanel("Services per Manager", plotOutput("managerPlot"))
      )
    )
  )
)

# ---------------------------------------------------
# Define Server Logic (How Data is Processed)
# ---------------------------------------------------
server <- function(input, output) {
  
  # ---------------------------------------------------
  # Histogram: Count of Services by Regional Variable
  # ---------------------------------------------------
  output$histogramPlot <- renderPlot({
    ggplot(directory_data, aes_string(x = input$regionVar, fill = input$groupVar)) +
      geom_bar(position = "stack") +  # Stacked bars to show count distribution
      theme_minimal() +
      labs(title = "Count of Services by Regional Variables", x = "Region", y = "Count", fill = "Grouping") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability
  })
  
  # ---------------------------------------------------
  # Bar Chart: Total Roll by Race, Stacked by Regional Variable
  # ---------------------------------------------------
  output$racePlot <- renderPlot({
    
    # Summarize total roll count per race category
    race_data <- directory_data %>%
      group_by(.data[[input$regionVar]]) %>%
      summarise(
        european_pakeha = sum(european_pakeha, na.rm = TRUE),
        maori = sum(maori, na.rm = TRUE),
        pacific = sum(pacific, na.rm = TRUE),
        asian = sum(asian, na.rm = TRUE),
        other = sum(other, na.rm = TRUE)
      ) %>%
      tidyr::pivot_longer(cols = -1, names_to = "Race", values_to = "Total")  # Convert to long format
    
    # Create a stacked bar chart
    ggplot(race_data, aes(x = .data[[input$regionVar]], y = Total, fill = Race)) +
      geom_bar(stat = "identity", position = "stack") +
      theme_minimal() +
      labs(title = "Total Roll by Race", x = "Region", y = "Total Roll", fill = "Race") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  # ---------------------------------------------------
  # Bar Chart: Total Roll by EQI Value
  # ---------------------------------------------------
  output$eqiPlot <- renderPlot({
    ggplot(directory_data, aes(x = equity_index_eqi, y = total, fill = equity_index_eqi)) +
      geom_bar(stat = "identity") +
      theme_minimal() +
      labs(title = "Total Roll by EQI Value", x = "EQI Value", y = "Total Roll") +
      theme(legend.position = "none")  # Hide legend since color represents EQI directly
  })
  
  # ---------------------------------------------------
  # Histogram: Count of Services per Manager
  # ---------------------------------------------------
  output$managerPlot <- renderPlot({
    
    # Count number of services per manager
    manager_data <- directory_data %>%
      count(management_contact_name)
    
    # Create a histogram of manager service counts
    ggplot(manager_data, aes(x = n)) +
      geom_histogram(bins = input$bins, fill = "blue", color = "black") +
      theme_minimal() +
      labs(title = "Count of Services per Manager", x = "Number of Services", y = "Frequency")
  })
}

# ---------------------------------------------------
# Run the Shiny App
# ---------------------------------------------------
shinyApp(ui = ui, server = server)
