**This notebook is an exercise in the [Geospatial Analysis](https://www.kaggle.com/learn/geospatial-analysis) course.  You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/your-first-map).**

---


# Introduction

[Kiva.org](https://www.kiva.org/) is an online crowdfunding platform extending financial services to poor people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people.

<center>
<img src="https://storage.googleapis.com/kaggle-media/learn/images/2G8C53X.png" width="500"><br/>
</center>

Kiva reaches some of the most remote places in the world through their global network of "Field Partners". These partners are local organizations working in communities to vet borrowers, provide services, and administer loans.

In this exercise, you'll investigate Kiva loans in the Philippines.  Can you identify regions that might be outside of Kiva's current network, in order to identify opportunities for recruiting new Field Partners?

To get started, run the code cell below to set up our feedback system.


```python
import geopandas as gpd
from matplotlib import pyplot as plt
```

### 1) Get the data.

Use the next cell to load the shapefile located at `loans_filepath` to create a GeoDataFrame `world_loans`.  


```python
# Your code here: Load the data
world_loans = gpd.read_file(r"C:\Users\LG PC\Desktop\data_mining\archive\kiva_loans\kiva_loans\kiva_loans.shp")

# Uncomment to view the first five rows of the data
world_loans.head()
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
      <th>Partner ID</th>
      <th>Field Part</th>
      <th>sector</th>
      <th>Loan Theme</th>
      <th>country</th>
      <th>amount</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>KREDIT Microfinance Institution</td>
      <td>General Financial Inclusion</td>
      <td>Higher Education</td>
      <td>Cambodia</td>
      <td>450</td>
      <td>POINT (102.89751 13.66726)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>KREDIT Microfinance Institution</td>
      <td>General Financial Inclusion</td>
      <td>Vulnerable Populations</td>
      <td>Cambodia</td>
      <td>20275</td>
      <td>POINT (102.98962 13.02870)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>KREDIT Microfinance Institution</td>
      <td>General Financial Inclusion</td>
      <td>Higher Education</td>
      <td>Cambodia</td>
      <td>9150</td>
      <td>POINT (102.98962 13.02870)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>KREDIT Microfinance Institution</td>
      <td>General Financial Inclusion</td>
      <td>Vulnerable Populations</td>
      <td>Cambodia</td>
      <td>604950</td>
      <td>POINT (105.31312 12.09829)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>KREDIT Microfinance Institution</td>
      <td>General Financial Inclusion</td>
      <td>Sanitation</td>
      <td>Cambodia</td>
      <td>275</td>
      <td>POINT (105.31312 12.09829)</td>
    </tr>
  </tbody>
</table>
</div>



### 2) Plot the data.

Run the next code cell without changes to load a GeoDataFrame `world` containing country boundaries.


```python
# This dataset is provided in GeoPandas
world_filepath = gpd.datasets.get_path('naturalearth_lowres')
world = gpd.read_file(world_filepath)
world.head()
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
      <th>0</th>
      <td>889953.0</td>
      <td>Oceania</td>
      <td>Fiji</td>
      <td>FJI</td>
      <td>5496</td>
      <td>MULTIPOLYGON (((180.00000 -16.06713, 180.00000...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58005463.0</td>
      <td>Africa</td>
      <td>Tanzania</td>
      <td>TZA</td>
      <td>63177</td>
      <td>POLYGON ((33.90371 -0.95000, 34.07262 -1.05982...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>603253.0</td>
      <td>Africa</td>
      <td>W. Sahara</td>
      <td>ESH</td>
      <td>907</td>
      <td>POLYGON ((-8.66559 27.65643, -8.66512 27.58948...</td>
    </tr>
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
  </tbody>
</table>
</div>



Use the `world` and `world_loans` GeoDataFrames to visualize Kiva loan locations across the world.


```python
# Your code here
ax = world.plot(figsize=(30,30), color=(0.7, 0.7, 0.7), edgecolor=(0.3, 0.3, 0.3))
world_loans.plot(ax=ax)
```




    <Axes: >




    
![png](output_8_1.png)
    


### 3) Select loans based in the Philippines.

Next, you'll focus on loans that are based in the Philippines.  Use the next code cell to create a GeoDataFrame `PHL_loans` which contains all rows from `world_loans` with loans that are based in the Philippines.


```python
Phil=world_loans['country']=="Philippines"
PHL_loans = world_loans[Phil].copy()
PHL_loans.head()
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
      <th>Partner ID</th>
      <th>Field Part</th>
      <th>sector</th>
      <th>Loan Theme</th>
      <th>country</th>
      <th>amount</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2859</th>
      <td>123</td>
      <td>Alalay sa Kaunlaran (ASKI)</td>
      <td>General Financial Inclusion</td>
      <td>General</td>
      <td>Philippines</td>
      <td>400</td>
      <td>POINT (121.73961 17.64228)</td>
    </tr>
    <tr>
      <th>2860</th>
      <td>123</td>
      <td>Alalay sa Kaunlaran (ASKI)</td>
      <td>General Financial Inclusion</td>
      <td>General</td>
      <td>Philippines</td>
      <td>400</td>
      <td>POINT (121.74169 17.63235)</td>
    </tr>
    <tr>
      <th>2861</th>
      <td>123</td>
      <td>Alalay sa Kaunlaran (ASKI)</td>
      <td>General Financial Inclusion</td>
      <td>General</td>
      <td>Philippines</td>
      <td>400</td>
      <td>POINT (121.46667 16.60000)</td>
    </tr>
    <tr>
      <th>2862</th>
      <td>123</td>
      <td>Alalay sa Kaunlaran (ASKI)</td>
      <td>General Financial Inclusion</td>
      <td>General</td>
      <td>Philippines</td>
      <td>6050</td>
      <td>POINT (121.73333 17.83333)</td>
    </tr>
    <tr>
      <th>2863</th>
      <td>123</td>
      <td>Alalay sa Kaunlaran (ASKI)</td>
      <td>General Financial Inclusion</td>
      <td>General</td>
      <td>Philippines</td>
      <td>625</td>
      <td>POINT (121.51800 16.72368)</td>
    </tr>
  </tbody>
</table>
</div>



### 4) Understand loans in the Philippines.

Run the next code cell without changes to load a GeoDataFrame `PHL` containing boundaries for all islands in the Philippines.


```python
# Load a KML file containing island boundaries
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
PHL = gpd.read_file(r"C:\Users\LG PC\Desktop\data_mining\archive\Philippines_AL258.kml", driver='KML')
PHL.head()
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
      <th>Name</th>
      <th>Description</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Autonomous Region in Muslim Mindanao</td>
      <td></td>
      <td>MULTIPOLYGON (((119.46690 4.58718, 119.46653 4...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bicol Region</td>
      <td></td>
      <td>MULTIPOLYGON (((124.04577 11.57862, 124.04594 ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cagayan Valley</td>
      <td></td>
      <td>MULTIPOLYGON (((122.51581 17.04436, 122.51568 ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Calabarzon</td>
      <td></td>
      <td>MULTIPOLYGON (((120.49202 14.05403, 120.49201 ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Caraga</td>
      <td></td>
      <td>MULTIPOLYGON (((126.45401 8.24400, 126.45407 8...</td>
    </tr>
  </tbody>
</table>
</div>



Use the `PHL` and `PHL_loans` GeoDataFrames to visualize loans in the Philippines.


```python
ax_1 = PHL.plot(figsize=(20,20), color=(0.7, 0.7, 0.7), edgecolor=(0.3, 0.3, 0.3))
PHL_loans.plot(ax=ax_1,alpha=0.5)
```




    <Axes: >




    
![png](output_14_1.png)
    


Can you identify any islands where it might be useful to recruit new Field Partners?  Do any islands currently look outside of Kiva's reach?

You might find [this map](https://bit.ly/2U2G7x7) useful to answer the question.

# Keep going

Continue to learn about **[coordinate reference systems](https://www.kaggle.com/alexisbcook/coordinate-reference-systems)**.

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/geospatial-analysis/discussion) to chat with other learners.*
