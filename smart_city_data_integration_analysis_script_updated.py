
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.regression.linear_model import OLS

# Load the dataset
df_smart_city = pd.read_excel('smart_city_data_integration_data_v2.xlsx')

# Step 1: Clustering Analysis
clustering_data = df_smart_city[['Traffic_Congestion', 'Public_Service_Usage', 'Pollution_Levels']]
kmeans = KMeans(n_clusters=3, random_state=42)
df_smart_city['Day_Cluster'] = kmeans.fit_predict(clustering_data)

# Step 2: Anomaly Detection using Isolation Forest on pollution levels
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df_smart_city['Anomaly'] = iso_forest.fit_predict(df_smart_city[['Pollution_Levels']])

# Step 3: Regression Analysis: Relationship between traffic congestion and public service usage
X = df_smart_city[['Traffic_Congestion', 'Temperature']]
y = df_smart_city['Public_Service_Usage']
model_ols = OLS(y, X).fit()

# Visualization: Geographic Heatmap (Simulated data with random lat/lon)
np.random.seed(42)
latitudes = np.random.uniform(low=40.5, high=40.9, size=len(df_smart_city))
longitudes = np.random.uniform(low=-74.0, high=-73.5, size=len(df_smart_city))

plt.figure(figsize=(10,8))
plt.scatter(longitudes, latitudes, c=df_smart_city['Pollution_Levels'], cmap='RdYlBu', s=50, edgecolor='black')
plt.colorbar(label='Pollution Levels (PM2.5)')
plt.title('Geographic Heatmap: Pollution Levels')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('geographic_heatmap_pollution.png')
plt.close()

# Visualization: Time Series Plot of Traffic Congestion and Public Service Usage
plt.figure(figsize=(10,6))
plt.plot(df_smart_city['Date'], df_smart_city['Traffic_Congestion'], label='Traffic Congestion', color='blue')
plt.plot(df_smart_city['Date'], df_smart_city['Public_Service_Usage'], label='Public Service Usage', color='green')
plt.title('Time Series: Traffic Congestion vs Public Service Usage')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.legend()
plt.savefig('time_series_traffic_public_service.png')
plt.close()

# Visualization: Cluster Analysis
plt.figure(figsize=(10,6))
sns.scatterplot(x='Traffic_Congestion', y='Pollution_Levels', hue='Day_Cluster', data=df_smart_city, palette='viridis', s=100)
plt.title('Cluster Analysis: Traffic Congestion vs Pollution Levels')
plt.xlabel('Traffic Congestion')
plt.ylabel('Pollution Levels')
plt.savefig('cluster_analysis_traffic_pollution.png')
plt.close()

# Visualization: Anomaly Detection in Pollution Levels
plt.figure(figsize=(10,6))
sns.scatterplot(x='Date', y='Pollution_Levels', hue='Anomaly', data=df_smart_city, palette='Set1', s=100)
plt.title('Anomaly Detection: Pollution Levels')
plt.xlabel('Date')
plt.ylabel('Pollution Levels')
plt.savefig('anomaly_detection_pollution.png')
plt.close()

# Display OLS regression summary
print(model_ols.summary())
