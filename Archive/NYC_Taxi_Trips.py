#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import IPython.display as display
import matplotlib.pyplot as plt 
import datetime
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split   # This is to prepare the data for the model
from sklearn.tree import DecisionTreeClassifier 
# heatmap
get_ipython().system('pip install folium ')
import folium
from folium import plugins


# Importing the raw data- 2.34 GB

# In[ ]:


df=pd.read_csv(r'nyc_taxi_data_2014.csv', low_memory=False)


# In[ ]:


df.head(5)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isnull().any().any()


# no nulls at all...<br>
# let's double check

# In[ ]:


df.isnull().sum().sum()


# In[ ]:


df.shape


# let's get to know each column by it-self

# Describing the entire DF

# In[ ]:


df.describe(include='all')


# dropping unnecessary colummns: <br>
# 1. mta_tax- 0.50 dollar MTA tax that is automatically triggered based on the metered rate in use, is the same for all rows<br>
# 2. Store_and_fwd_flag- This flag indicates whether the trip record was held in vehicle memory before sending to the vendor, aka “store and forward,”because the vehicle did not have a connection to the server. <br> 
# Y= store and forward trip , N= not a store and forward trip <br>
# 3. surcharge- Miscellaneous extras and surcharges. Currently, this only includes the 0.50 dollar and 1 dollar rush hour and overnight charges. <br>
# 4. Tolls_amount- Total amount of all tolls paid in trip by the passenger. <br>

# In[ ]:


df=df.drop(['mta_tax','surcharge','store_and_fwd_flag','tolls_amount'], axis=1)


# In[ ]:


df.shape


# In[ ]:


smallDF = df.sample(frac =.1)


# In[ ]:


smallDF.shape


# In[ ]:


smallDF.to_csv("smaller_dataframe_taxi_trips", sep='\t')


# ###### The Project will start from that point in order to save time of loading the data next time.<br>
# We use "only" 10% of the raw data which equals to 1.5M rows

# In[2]:


data=pd.read_csv(r'smaller_dataframe_taxi_trips',sep='\t',index_col='Unnamed: 0')


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data=data.dropna(axis=0)
data.info()


# In[43]:


data.head()


# ### Detecting outlier trips (outside of NYC) and remove them

# In[23]:


xlim = [-74.03, -73.77] #taken from Kaggle discussion
ylim = [40.63, 40.85] #taken from the same Kaggle discussion
data = data[(data.pickup_longitude> xlim[0]) & (data.pickup_longitude < xlim[1])]
data = data[(data.dropoff_longitude> xlim[0]) & (data.dropoff_longitude < xlim[1])]
data = data[(data.pickup_latitude> ylim[0]) & (data.pickup_latitude < ylim[1])]
data = data[(data.dropoff_latitude> ylim[0]) & (data.dropoff_latitude < ylim[1])]


# pickup locations

# In[30]:


plt.plot(data['pickup_longitude'], data['pickup_latitude'], '.', color='gray', alpha=0.1)
plt.title('Pickup Location Lat & Long', weight = 'bold')
plt.show()


# Dropoff Locations

# In[36]:


plt.plot(data['dropoff_longitude'], data['dropoff_latitude'], '.', color='black', alpha=0.07)
plt.title('Dropoff Location Lat & Long', weight = 'bold')
plt.show()


# #### Transforming time format and calculating trip duration

# In[50]:


data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], format= '%Y/%m/%d %H:%M')
data['dropoff_datetime']= pd.to_datetime(data['dropoff_datetime'], format= '%Y/%m/%d %H:%M')
data['trip_duration_min'] =  data.dropoff_datetime -data.pickup_datetime
#sd


# #### Extracting Hour, Day of the Week and Month

# In[10]:


data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data["pickup_day"] = data["pickup_datetime"].dt.strftime('%u').astype(int)
data["pickup_hour"] = data["pickup_datetime"].dt.strftime('%H').astype(int)
data["pickup_month"] = data["pickup_datetime"].dt.strftime('%m').astype(int)


# In[21]:


data.head(5)


# weekdays

# In[12]:


weekday_dict = {1: "Mon",
                       2: "Tues",
                       3: "Wed",
                       4: "Thurs",
                       5: "Fri",
                       6: "Sat",
                       7: "Sun"}
data['weekday']=data['pickup_day'].map(weekday_dict)


# In[13]:


month_dict = {1: "Jan",
                       2: "Feb",
                       3: "March",
                       4: "April",
                       5: "May",
                       6: "June",
                       7:"July",
                       8:"Aug",
                       9:"Sep",
                       10:"Oct",
                       11:"Nov",
                       12:"Dec"}
data['month']=data['pickup_month'].map(month_dict)


# In[56]:


weekday_list = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']
g = sns.factorplot(kind='bar',        # Boxplot
               y='trip_duration_min',       # Y-axis - values for boxplot
               x='weekday',        # X-axis - first factor
               #estimator = np.sum, 
               data=data,        # Dataframe 
               size=4,            # Figure size (x100px)      
               aspect=1.6,        # Width = size * aspect 
               order = list(weekday_list),
               legend_out=False) 
plt.title('Avg Trip Durations by Weekday\n', weight = 'bold', size = 15)
plt.xlabel('Weekday', size = 12,weight = 'bold')
plt.ylabel('Average trip duration', size = 12,weight = 'bold')
g.set_xticklabels(rotation=45)


# #### Scatter- Heatmap 

# pickup location

# In[41]:


## Pickup Data
nyc_map_pickup = folium.Map([40.7306,-73.935242], zoom_start=11)

for index, row in data.iterrows():
    folium.CircleMarker([row['pickup_latitude'], row['pickup_longitude']],
                        radius=1,
                        fill=True,
                        fill_color="blue", 
                        fill_opacity=0.005,
                       )#.add_to(nyc_map) if you want to add points/markers to the map - does not look well with heatmaps


# In[ ]:


# convert to (n, 2) nd-array format for heatmap
pickupArr = data[['pickup_latitude', 'pickup_longitude']].as_matrix()

# plot heatmap
nyc_map_pickup.add_child(plugins.HeatMap(pickupArr, radius=15))


# dropoff locations

# In[ ]:


## Dropoff Data
nyc_map_dropoff = folium.Map([40.7306,-73.935242], zoom_start=11)

for index, row in df.iterrows():
    folium.CircleMarker([row['dropoff_latitude'], row['dropoff_longitude']],
                        radius=1,
                        fill=True,
                        fill_color="blue", 
                        fill_opacity=0.005,
                       )#.add_to(nyc_map) if you want to add points/markers to the map - does not look well with heatmaps


# In[ ]:


dropoffArr = df[['dropoff_latitude', 'dropoff_longitude']].as_matrix()

# plot heatmap
nyc_map_dropoff.add_child(plugins.HeatMap(dropoffArr, radius=15))


# # EDA

# weekdays first

# In[ ]:


weekday_list = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']
g = sns.factorplot(kind='bar',        # Boxplot
               y='trip_duration',       # Y-axis - values for boxplot
               x='weekday',        # X-axis - first factor
               #estimator = np.sum, 
               data=subset_train,        # Dataframe 
               size=6,            # Figure size (x100px)      
               aspect=1.6,        # Width = size * aspect 
               order = list(weekday_list),
               legend_out=False) 
plt.title('Avg Trip Durations by Weekday\n', weight = 'bold', size = 20)
plt.xlabel('Weekday', size = 18,weight = 'bold')
plt.ylabel('Average trip duration', size = 18,weight = 'bold')
g.set_xticklabels(rotation=45)

