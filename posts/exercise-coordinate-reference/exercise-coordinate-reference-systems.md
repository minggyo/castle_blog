---
title: "coordinate-reference"
date: "2023-05-31"
categories: [code, analysis]
---

**This notebook is an exercise in the [Geospatial Analysis](https://www.kaggle.com/learn/geospatial-analysis) course.  You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/coordinate-reference-systems).**

---


# Introduction

You are a bird conservation expert and want to understand migration patterns of purple martins.  In your research, you discover that these birds typically spend the summer breeding season in the eastern United States, and then migrate to South America for the winter.  But since this bird is under threat of endangerment, you'd like to take a closer look at the locations that these birds are more likely to visit.

<center>
<img src="https://storage.googleapis.com/kaggle-media/learn/images/qQcS0KM.png" width="1000"><br/>
</center>

There are several [protected areas](https://www.iucn.org/theme/protected-areas/about) in South America, which operate under special regulations to ensure that species that migrate (or live) there have the best opportunity to thrive.  You'd like to know if purple martins tend to visit these areas.  To answer this question, you'll use some recently collected data that tracks the year-round location of eleven different birds.

Before you get started, run the code cell below to set everything up.


```python
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString
```

# Exercises

### 1) Load the data.

Run the next code cell (without changes) to load the GPS data into a pandas DataFrame `birds_df`.  


```python
# Load the data and print the first 5 rows
birds_df = pd.read_csv(r"C:\Users\LG PC\Desktop\data_mining\archive\purple_martin.csv", parse_dates=['timestamp'])
print("There are {} different birds in the dataset.".format(birds_df["tag-local-identifier"].nunique()))
birds_df.head()
```

    There are 11 different birds in the dataset.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>location-long</th>
      <th>location-lat</th>
      <th>tag-local-identifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-15 05:56:00</td>
      <td>-88.146014</td>
      <td>17.513049</td>
      <td>30448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-09-01 05:59:00</td>
      <td>-85.243501</td>
      <td>13.095782</td>
      <td>30448</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-10-30 23:58:00</td>
      <td>-62.906089</td>
      <td>-7.852436</td>
      <td>30448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-11-15 04:59:00</td>
      <td>-61.776826</td>
      <td>-11.723898</td>
      <td>30448</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-11-30 09:59:00</td>
      <td>-61.241538</td>
      <td>-11.612237</td>
      <td>30448</td>
    </tr>
  </tbody>
</table>
</div>



There are 11 birds in the dataset, where each bird is identified by a unique value in the "tag-local-identifier" column.  Each bird has several measurements, collected at different times of the year.

Use the next code cell to create a GeoDataFrame `birds`.  
- `birds` should have all of the columns from `birds_df`, along with a "geometry" column that contains Point objects with (longitude, latitude) locations.  
- Set the CRS of `birds` to `{'init': 'epsg:4326'}`.


```python
birds_df = birds_df.rename(columns={'location-long': 'longitude', 'location-lat': 'latitude'})
birds_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>tag-local-identifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-15 05:56:00</td>
      <td>-88.146014</td>
      <td>17.513049</td>
      <td>30448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-09-01 05:59:00</td>
      <td>-85.243501</td>
      <td>13.095782</td>
      <td>30448</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-10-30 23:58:00</td>
      <td>-62.906089</td>
      <td>-7.852436</td>
      <td>30448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-11-15 04:59:00</td>
      <td>-61.776826</td>
      <td>-11.723898</td>
      <td>30448</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-11-30 09:59:00</td>
      <td>-61.241538</td>
      <td>-11.612237</td>
      <td>30448</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Your code here: Create the GeoDataFrame
birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df.longitude, birds_df.latitude))

# Your code here: Set the CRS to {'init': 'epsg:4326'}
birds.crs = {'init': 'epsg:4326'}
```

    c:\Users\LG PC\anaconda3\envs\min\Lib\site-packages\pyproj\crs\crs.py:141: FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
      in_crs_string = _prepare_from_proj_string(in_crs_string)
    


```python
birds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>tag-local-identifier</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-15 05:56:00</td>
      <td>-88.146014</td>
      <td>17.513049</td>
      <td>30448</td>
      <td>POINT (-88.14601 17.51305)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-09-01 05:59:00</td>
      <td>-85.243501</td>
      <td>13.095782</td>
      <td>30448</td>
      <td>POINT (-85.24350 13.09578)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-10-30 23:58:00</td>
      <td>-62.906089</td>
      <td>-7.852436</td>
      <td>30448</td>
      <td>POINT (-62.90609 -7.85244)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-11-15 04:59:00</td>
      <td>-61.776826</td>
      <td>-11.723898</td>
      <td>30448</td>
      <td>POINT (-61.77683 -11.72390)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-11-30 09:59:00</td>
      <td>-61.241538</td>
      <td>-11.612237</td>
      <td>30448</td>
      <td>POINT (-61.24154 -11.61224)</td>
    </tr>
  </tbody>
</table>
</div>



### 2) Plot the data.

Next, we load in the `'naturalearth_lowres'` dataset from GeoPandas, and set `americas` to a GeoDataFrame containing the boundaries of all countries in the Americas (both North and South America).  Run the next code cell without changes.


```python
# Load a GeoDataFrame with country boundaries in North/South America, print the first 5 rows
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
americas = world.loc[world['continent'].isin(['North America', 'South America'])]
americas.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pop_est</th>
      <th>continent</th>
      <th>name</th>
      <th>iso_a3</th>
      <th>gdp_md_est</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>37589262.0</td>
      <td>North America</td>
      <td>Canada</td>
      <td>CAN</td>
      <td>1736425</td>
      <td>MULTIPOLYGON (((-122.84000 49.00000, -122.9742...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>328239523.0</td>
      <td>North America</td>
      <td>United States of America</td>
      <td>USA</td>
      <td>21433226</td>
      <td>MULTIPOLYGON (((-122.84000 49.00000, -120.0000...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>44938712.0</td>
      <td>South America</td>
      <td>Argentina</td>
      <td>ARG</td>
      <td>445445</td>
      <td>MULTIPOLYGON (((-68.63401 -52.63637, -68.25000...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>18952038.0</td>
      <td>South America</td>
      <td>Chile</td>
      <td>CHL</td>
      <td>282318</td>
      <td>MULTIPOLYGON (((-68.63401 -52.63637, -68.63335...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>11263077.0</td>
      <td>North America</td>
      <td>Haiti</td>
      <td>HTI</td>
      <td>14332</td>
      <td>POLYGON ((-71.71236 19.71446, -71.62487 19.169...</td>
    </tr>
  </tbody>
</table>
</div>



Use the next code cell to create a single plot that shows both: (1) the country boundaries in the `americas` GeoDataFrame, and (2) all of the points in the `birds_gdf` GeoDataFrame.  

Don't worry about any special styling here; just create a preliminary plot, as a quick sanity check that all of the data was loaded properly.  In particular, you don't have to worry about color-coding the points to differentiate between birds, and you don't have to differentiate starting points from ending points.  We'll do that in the next part of the exercise.


```python
# Your code here
ax = americas.plot(figsize=(10,10), color='whitesmoke', edgecolor='black')
birds.plot(markersize=3, ax=ax)
```




    <Axes: >




    
![png](output_12_1.png)
    


### 3) Where does each bird start and end its journey? (Part 1)

Now, we're ready to look more closely at each bird's path.  Run the next code cell to create two GeoDataFrames:
- `path_gdf` contains LineString objects that show the path of each bird.  It uses the `LineString()` method to create a LineString object from a list of Point objects.
- `start_gdf` contains the starting points for each bird.


```python
# GeoDataFrame showing path for each bird
path_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: LineString(x)).reset_index()
path_gdf = gpd.GeoDataFrame(path_df, geometry=path_df.geometry)
path_gdf.crs = {'init' :'epsg:4326'}

# GeoDataFrame showing starting point for each bird
start_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[0]).reset_index()
start_gdf = gpd.GeoDataFrame(start_df, geometry=start_df.geometry)
start_gdf.crs = {'init' :'epsg:4326'}

# Show first five rows of GeoDataFrame
start_gdf.head()
```

    c:\Users\LG PC\anaconda3\envs\min\Lib\site-packages\pyproj\crs\crs.py:141: FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
      in_crs_string = _prepare_from_proj_string(in_crs_string)
    c:\Users\LG PC\anaconda3\envs\min\Lib\site-packages\pyproj\crs\crs.py:141: FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
      in_crs_string = _prepare_from_proj_string(in_crs_string)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tag-local-identifier</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30048</td>
      <td>POINT (-90.12992 20.73242)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30054</td>
      <td>POINT (-93.60861 46.50563)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30198</td>
      <td>POINT (-80.31036 25.92545)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30263</td>
      <td>POINT (-76.78146 42.99209)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30275</td>
      <td>POINT (-76.78213 42.99207)</td>
    </tr>
  </tbody>
</table>
</div>



Use the next code cell to create a GeoDataFrame `end_gdf` containing the final location of each bird.  
- The format should be identical to that of `start_gdf`, with two columns ("tag-local-identifier" and "geometry"), where the "geometry" column contains Point objects.
- Set the CRS of `end_gdf` to `{'init': 'epsg:4326'}`.


```python
# Your code here
end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index()
end_gdf = gpd.GeoDataFrame(start_df, geometry=end_df.geometry)
end_gdf.crs = {'init': 'epsg:4326'}

end_gdf.head()
```

    c:\Users\LG PC\anaconda3\envs\min\Lib\site-packages\pyproj\crs\crs.py:141: FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
      in_crs_string = _prepare_from_proj_string(in_crs_string)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tag-local-identifier</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30048</td>
      <td>POINT (-47.53632 -4.43758)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30054</td>
      <td>POINT (-62.47914 -5.03840)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30198</td>
      <td>POINT (-57.46417 -2.77617)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30263</td>
      <td>POINT (-50.19230 -5.70504)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30275</td>
      <td>POINT (-57.70404 -16.72336)</td>
    </tr>
  </tbody>
</table>
</div>



### 4) Where does each bird start and end its journey? (Part 2)

Use the GeoDataFrames from the question above (`path_gdf`, `start_gdf`, and `end_gdf`) to visualize the paths of all birds on a single map.  You may also want to use the `americas` GeoDataFrame.


```python
# Your code here
ax = americas.plot(figsize=(10,10), color='whitesmoke', edgecolor='black')
path_gdf.plot(linewidth=1, ax=ax)
start_gdf.plot(markersize=10, ax=ax, color="red", zorder=2)
end_gdf.plot(markersize=10, ax=ax, color="blue", zorder=2)
```




    <Axes: >




    
![png](output_18_1.png)
    


### 5) Where are the protected areas in South America? (Part 1)

It looks like all of the birds end up somewhere in South America.  But are they going to protected areas?

In the next code cell, you'll create a GeoDataFrame `protected_areas` containing the locations of all of the protected areas in South America.  The corresponding shapefile is located at filepath `protected_filepath`.


```python
# Path of the shapefile to load
protected_filepath = r"C:\Users\LG PC\Desktop\data_mining\archive\SAPA_Aug2019-shapefile\SAPA_Aug2019-shapefile\SAPA_Aug2019-shapefile-polygons.shp"

# Your code here
protected_areas = gpd.read_file(protected_filepath)
```


```python
protected_areas.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WDPAID</th>
      <th>WDPA_PID</th>
      <th>PA_DEF</th>
      <th>NAME</th>
      <th>ORIG_NAME</th>
      <th>DESIG</th>
      <th>DESIG_ENG</th>
      <th>DESIG_TYPE</th>
      <th>IUCN_CAT</th>
      <th>INT_CRIT</th>
      <th>...</th>
      <th>GOV_TYPE</th>
      <th>OWN_TYPE</th>
      <th>MANG_AUTH</th>
      <th>MANG_PLAN</th>
      <th>VERIF</th>
      <th>METADATAID</th>
      <th>SUB_LOC</th>
      <th>PARENT_ISO</th>
      <th>ISO3</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14067.0</td>
      <td>14067</td>
      <td>1</td>
      <td>Het Spaans Lagoen</td>
      <td>Het Spaans Lagoen</td>
      <td>Ramsar Site, Wetland of International Importance</td>
      <td>Ramsar Site, Wetland of International Importance</td>
      <td>International</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>...</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>Management plan is not implemented and not ava...</td>
      <td>State Verified</td>
      <td>1856</td>
      <td>Not Reported</td>
      <td>NLD</td>
      <td>ABW</td>
      <td>POLYGON ((-69.97523 12.47379, -69.97523 12.473...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14003.0</td>
      <td>14003</td>
      <td>1</td>
      <td>Bubali Pond Bird Sanctuary</td>
      <td>Bubali Pond Bird Sanctuary</td>
      <td>Bird Sanctuary</td>
      <td>Bird Sanctuary</td>
      <td>National</td>
      <td>Not Reported</td>
      <td>Not Applicable</td>
      <td>...</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>State Verified</td>
      <td>1899</td>
      <td>Not Reported</td>
      <td>NLD</td>
      <td>ABW</td>
      <td>POLYGON ((-70.04734 12.56329, -70.04615 12.563...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>555624439.0</td>
      <td>555624439</td>
      <td>1</td>
      <td>Arikok National Park</td>
      <td>Arikok National Park</td>
      <td>National Park</td>
      <td>National Park</td>
      <td>National</td>
      <td>Not Reported</td>
      <td>Not Applicable</td>
      <td>...</td>
      <td>Non-profit organisations</td>
      <td>Non-profit organisations</td>
      <td>Fundacion Parke Nacional Arikok</td>
      <td>Not Reported</td>
      <td>State Verified</td>
      <td>1899</td>
      <td>Not Reported</td>
      <td>NLD</td>
      <td>ABW</td>
      <td>MULTIPOLYGON (((-69.96302 12.48384, -69.96295 ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>303894.0</td>
      <td>303894</td>
      <td>1</td>
      <td>Madidi</td>
      <td>Madidi</td>
      <td>Area Natural de Manejo Integrado</td>
      <td>Natural Integrated Management Area</td>
      <td>National</td>
      <td>Not Reported</td>
      <td>Not Applicable</td>
      <td>...</td>
      <td>Federal or national ministry or agency</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>State Verified</td>
      <td>1860</td>
      <td>BO-L</td>
      <td>BOL</td>
      <td>BOL</td>
      <td>POLYGON ((-68.59060 -14.43388, -68.59062 -14.4...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>303893.0</td>
      <td>303893</td>
      <td>1</td>
      <td>Apolobamba</td>
      <td>Apolobamba</td>
      <td>Area Natural de Manejo Integado Nacional</td>
      <td>National Natural Integrated Management Area</td>
      <td>National</td>
      <td>Not Reported</td>
      <td>Not Applicable</td>
      <td>...</td>
      <td>Federal or national ministry or agency</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>Not Reported</td>
      <td>State Verified</td>
      <td>1860</td>
      <td>BO-L</td>
      <td>BOL</td>
      <td>BOL</td>
      <td>POLYGON ((-69.20949 -14.73334, -69.20130 -14.7...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



### 6) Where are the protected areas in South America? (Part 2)

Create a plot that uses the `protected_areas` GeoDataFrame to show the locations of the protected areas in South America.  (_You'll notice that some protected areas are on land, while others are in marine waters._)


```python
# Country boundaries in South America
south_america = americas.loc[americas['continent']=='South America']
# Your code here: plot protected areas in South America
ax = south_america.plot(figsize=(10,10), color='whitesmoke', edgecolor='black')
protected_areas.plot(ax=ax)
```




    <Axes: >




    
![png](output_23_1.png)
    


### 7) What percentage of South America is protected?

You're interested in determining what percentage of South America is protected, so that you know how much of South America is suitable for the birds.  

As a first step, you calculate the total area of all protected lands in South America (not including marine area).  To do this, you use the "REP_AREA" and "REP_M_AREA" columns, which contain the total area and total marine area, respectively, in square kilometers.

Run the code cell below without changes.


```python
P_Area = sum(protected_areas['REP_AREA']-protected_areas['REP_M_AREA'])
print("South America has {} square kilometers of protected areas.".format(P_Area))
```

    South America has 5396761.9116883585 square kilometers of protected areas.
    

Then, to finish the calculation, you'll use the `south_america` GeoDataFrame.  


```python
south_america.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pop_est</th>
      <th>continent</th>
      <th>name</th>
      <th>iso_a3</th>
      <th>gdp_md_est</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>44938712.0</td>
      <td>South America</td>
      <td>Argentina</td>
      <td>ARG</td>
      <td>445445</td>
      <td>MULTIPOLYGON (((-68.63401 -52.63637, -68.25000...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>18952038.0</td>
      <td>South America</td>
      <td>Chile</td>
      <td>CHL</td>
      <td>282318</td>
      <td>MULTIPOLYGON (((-68.63401 -52.63637, -68.63335...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3398.0</td>
      <td>South America</td>
      <td>Falkland Is.</td>
      <td>FLK</td>
      <td>282</td>
      <td>POLYGON ((-61.20000 -51.85000, -60.00000 -51.2...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3461734.0</td>
      <td>South America</td>
      <td>Uruguay</td>
      <td>URY</td>
      <td>56045</td>
      <td>POLYGON ((-57.62513 -30.21629, -56.97603 -30.1...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>211049527.0</td>
      <td>South America</td>
      <td>Brazil</td>
      <td>BRA</td>
      <td>1839758</td>
      <td>POLYGON ((-53.37366 -33.76838, -53.65054 -33.2...</td>
    </tr>
  </tbody>
</table>
</div>



Calculate the total area of South America by following these steps:
- Calculate the area of each country using the `area` attribute of each polygon (with EPSG 3035 as the CRS), and add up the results.  The calculated area will be in units of square meters.
- Convert your answer to have units of square kilometeters.


```python
# Your code here: Calculate the total area of South America (in square kilometers)
south_america=south_america.to_crs(epsg=3035)
totalArea = south_america.loc[:, "AREA"] = south_america.geometry.area / 10**6
```

Run the code cell below to calculate the percentage of South America that is protected.


```python
# What percentage of South America is protected?
percentage_protected = P_Area/totalArea
print('Approximately {}% of South America is protected.'.format(round(percentage_protected*100, 2)))
```

    Approximately 9        193.85
    10       663.08
    20     32963.98
    28      3050.55
    29        63.43
    30       497.39
    31       412.50
    32       468.51
    40       594.02
    41      2571.97
    42      3740.15
    44      2152.29
    156     1346.55
    dtype: float64% of South America is protected.
    

### 8) Where are the birds in South America?

So, are the birds in protected areas?  

Create a plot that shows for all birds, all of the locations where they were discovered in South America.  Also plot the locations of all protected areas in South America.

To exclude protected areas that are purely marine areas (with no land component), you can use the "MARINE" column (and plot only the rows in `protected_areas[protected_areas['MARINE']!='2']`, instead of every row in the `protected_areas` GeoDataFrame).


```python
# Your code here

```

# Keep going

Create stunning **[interactive maps](https://www.kaggle.com/alexisbcook/interactive-maps)** with your geospatial data.

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/geospatial-analysis/discussion) to chat with other learners.*
