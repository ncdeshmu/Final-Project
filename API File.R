# API File
#First, we have to install the following libraries:
library(plumber) #a package that allows you to create APIs by decorating the existing code
library(tidymodels)
library(dplyr)
library(ggplot2)
library(yardstick) #a package that is used to estimate how well models are working
library(readr)

#We will then be reading in the dataset, and then printing in the first set of rows
#of the dataset
diabetes_health_indicators <- read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
head(diabetes_health_indicators)

#Convert the variables to different levels or factors, using the factor() command
#For instance, we are converting the 0s and 1s, to "No" and "Yes" respectively
diabetes_health_indicators <- diabetes_health_indicators |>
  mutate(
     Diabetes_binary = factor(Diabetes_binary, levels = c(0,1), labels = c("No", "Yes")),
     HighBP = factor(HighBP, levels = c(0,1), labels = c("No", "Yes")),
     HighChol = factor(HighChol, levels = c(0,1), labels = c("No", "Yes")),
     CholCheck = factor(CholCheck, levels = c(0,1), labels = c("No", "Yes")),
     BMI = factor(BMI, levels = c(0,1), labels = c("No", "Yes")),
     Smoker = factor(Smoker, levels = c(0,1), labels = c("No", "Yes")),
     Stroke = factor(Stroke, levels = c(0,1), labels = c("No", "Yes")),
     HeartDiseaseorAttack = factor(HeartDiseaseorAttack, levels = c(0,1), labels = c("No", "Yes")),
     PhysActivity = factor(PhysActivity, levels = c(0,1), labels = c("No", "Yes")),
     Fruits = factor(Fruits, levels = c(0,1), labels = c("No", "Yes")),
     Veggies = factor(Veggies, levels = c(0,1), labels = c("No", "Yes")),
     HvyAlcoholConsump = factor(HvyAlcoholConsump, levels = c(0,1), labels = c("No", "Yes")),
     AnyHealthcare = factor(AnyHealthcare, levels = c(0,1), labels = c("No", "Yes")),
     NoDocbcCost = factor(NoDocbcCost, levels = c(0,1), labels = c("No", "Yes")),
     GenHlth = factor(GenHlth, levels = c(0,1), labels = c("No", "Yes")),
     MentHlth = factor(MentHlth, levels = c(0,1), labels = c("No", "Yes")),
     PhysHlth = factor(PhysHlth, levels = c(0,1), labels = c("No", "Yes")),
     DiffWalk = factor(DiffWalk, levels = c(0,1), labels = c("No", "Yes")),
     Sex = factor(Sex, levels = c(0,1), labels = c("Female", "Male")),
     Age = factor(Age, levels = c(1:13), labels = c("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+")),
     Education = factor(Education, levels = c(1:6), labels = c("Kindergarten or No School", "Elementary", "Middle school", "High school", "College","Professional Degree")),
     Income = factor(Income, levels = c(1:8), labels = c("<$10k", "$10k-$15k", "$15k-$20k", "$20k-$25k", "$25k-$35k", "$35k-$50k", "$50k-$75k", "$75k+"))
  )

#The BMI (Body Mass Index) variable will be converted to a numeric variable using
#the "as.numeric()" command.
#Additionally, we will also be checking for null values or "NA" values using the
#command the "is.na" command.
diabetes_health_indicators$BMI <- as.numeric(diabetes_health_indicators$BMI)
sum(is.na(diabetes_health_indicators$BMI))

#Recipe with Normalization
LR2_recipe <- recipe(Diabetes_binary ~ BMI + Smoker + HighBP + HeartDiseaseorAttack + PhysActivity + Income, 
                     data = diabetes_health_indicators) %>%
  step_normalize(BMI)
    
#We will then be creating a Random Forest Model with 1000 trees, along with 
#setting our engine, as well as mode. 
rf_spec <- rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

#Workflow
rf_wkf <- workflow() %>%
  add_recipe(LR2_recipe) %>%
  add_model(rf_spec)
  
#Fit the Model for the Entire Dataset
library(ranger)
final_model <- fit(rf_wkf, data = diabetes_health_indicators)

#Default values for the predictor variables

#Endpoints:
#As stated in the project requirements, we are to have three different endpoints:
#Prediction Endpoint, Info Endpoint, and Confusion Matrix Endpoint
#We will begin by creating a "Prediction endpoint"

#Prediction endpoint
#This prediction endpoint will be taking in some predictor variables as the parameters,
#as indicated by the "@param" sign:
#* @param HighBP Blood pressure status
#* @param BMI Body Mass Index
#* @param Smoker Smoking status
#* @param HeartDiseaseorAttack Heart disease status
#* @param PhysActivity Physical activity status
#* @param Income Income
#* @get /pred
function(HighBP = default_values$HighBP,
         BMI = default_values$BMI,
         Smoker = default_values$Smoker,
         HeartDiseaseorAttack = default_values$HeartDiseaseorAttack,
         PhysActivity = default_values$PhysActivity,
         Income = default_values$Income) {

#Creating a tibble
newdataframe <- tibble(
  HighBP = factor(HighBP, levels = c("No", "Yes")),
  BMI = as.numeric(BMI),
  Smoker = factor(Smoker, levels = c("No", "Yes")),
  HeartDiseaseorAttack = factor(HeartDiseaseorAttack, levels = c("No", "Yes")),
  PhysActivity = factor(PhysActivity, levels = c("No", "Yes")),
  Income = factor(Income, levels = c(1:8), labels = c("<$10k", "$10k-$15k", "$15k-$20k", "$20k-$25k", "$25k-$35k", "$35k-$50k", "$50k-$75k", "$75k+"))
)


#Info endpoint
#* @get /info
function() {
  return(list(
    name = "Diabetes Prediction API",
    url = "https://github.com/ncdeshmu/Final-Project.git"
  ))
}

#Confusion matrix endpoint (we are trying to return a "png" image through 
"@serializer"
#* @get /confusion
#* @serializer png
function() {
  # Take in the predictions for the entire dataset and ensure factors are used
  predictions <- predict(final_rf_model, diabetes_health_indicators) %>%
# Create a confusion matrix, based on the predictions
cm <- conf_mat(predictions, truth = Diabetes_binary, estimate = .pred_class)

# Plot the confusion matrix
cm_plot <- autoplot(cm)

# Return the plot as PNG
cm_plot
}

#Create and run the Plumber endpoints
plumber <- Plumber$new()
pr <- Plumber$new()
#pr$print()
pr$handle("GET", "/pred", pred_endpoint)
pr$handle("GET", "/info", info_endpoint)
pr$handle("GET", "/confusion", confusion_endpoint)

# Run the API server
pr$run(port = 8000, swagger = TRUE)