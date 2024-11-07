
# Load necessary libraries
library(tidyverse)
library(cluster)
library(factoextra)
library(caret)
library(randomForest)
library(ggplot2)

# Load the dataset
df_smart_city <- read.csv("smart_city_data_integration_data_v2.csv")

# Step 1: Clustering Analysis
set.seed(42)
kmeans_result <- kmeans(df_smart_city[, c('Traffic_Congestion', 'Public_Service_Usage', 'Pollution_Levels')], centers = 3, nstart = 25)
df_smart_city$Day_Cluster <- kmeans_result$cluster

# Visualization: Cluster Analysis
ggplot(df_smart_city, aes(x = Traffic_Congestion, y = Pollution_Levels, color = as.factor(Day_Cluster))) +
  geom_point(size = 3) +
  labs(title = "Cluster Analysis: Traffic Congestion vs Pollution Levels", x = "Traffic Congestion", y = "Pollution Levels", color = "Cluster") +
  theme_minimal()

# Step 2: Anomaly Detection using Isolation Forest
iso_forest_model <- randomForest(Pollution_Levels ~ ., data = df_smart_city, proximity = TRUE)
df_smart_city$Anomaly <- ifelse(iso_forest_model$predicted > quantile(iso_forest_model$predicted, 0.95), 1, 0)

# Visualization: Anomaly Detection
ggplot(df_smart_city, aes(x = Date, y = Pollution_Levels, color = as.factor(Anomaly))) +
  geom_point(size = 3) +
  labs(title = "Anomaly Detection: Pollution Levels", x = "Date", y = "Pollution Levels", color = "Anomaly") +
  theme_minimal()

# Step 3: Regression Analysis: Relationship between traffic congestion and public service usage
model_ols <- lm(Public_Service_Usage ~ Traffic_Congestion + Temperature, data = df_smart_city)
summary(model_ols)

# Visualization: Time Series Plot of Traffic Congestion and Public Service Usage
ggplot(df_smart_city, aes(x = Date)) +
  geom_line(aes(y = Traffic_Congestion, color = "Traffic Congestion")) +
  geom_line(aes(y = Public_Service_Usage, color = "Public Service Usage")) +
  labs(title = "Time Series: Traffic Congestion vs Public Service Usage", x = "Date", y = "Index Value", color = "Legend") +
  theme_minimal()

# Visualization: Geographic Heatmap
# Simulated data for lat/lon
set.seed(42)
df_smart_city$Latitude <- runif(nrow(df_smart_city), min = 40.5, max = 40.9)
df_smart_city$Longitude <- runif(nrow(df_smart_city), min = -74.0, max = -73.5)

ggplot(df_smart_city, aes(x = Longitude, y = Latitude, color = Pollution_Levels)) +
  geom_point(size = 3) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Geographic Heatmap: Pollution Levels", x = "Longitude", y = "Latitude", color = "Pollution Levels") +
  theme_minimal()
