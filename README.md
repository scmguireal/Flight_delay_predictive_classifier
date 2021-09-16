# Flight Delay Predictions 1-Week in Advance


The goal is to predict arrival delays of commercial flights without use of same day data. Often, there isn't much airlines can do to avoid the delays, therefore, they play an important role in both profits and loss of the airlines. It is critical for airlines to estimate flight delays as accurate as possible because the results can be applied to both, improvements in customer satisfaction and income of airline agencies.

### Data

We will be working with data from air travel industry. We will have three separate tables:

1. **flights**: The departure and arrival information about flights in US in years 2018 and 2019.
2. **fuel_comsumption**: The fuel comsumption of different airlines from years 2015-2019 aggregated per month.
3. **passengers**: The passenger totals on different routes from years 2015-2019 aggregated per month.


The data are stored in the Postgres database.


### Problem Statement

Predict whether any given flight will be late at any given time from the millions of flights in this database.

* Right so this is simple enough...



### The Approach

This repository is only focused on traditional ML techniques. I recognize that with the amount of data present NN would be a viable option and I may update this in the future to include them as well, but for now it's all about ML.

The sheer amount of data included in this database is too much to process without more hardware power, so we decided to sample the data.

We decided on taking the top 10 airports as our sample as it brought number of flights down to about 4M and included enough information across our big three features in the hypothesis:
* Geographic Location
* Airline Carrier
* Time of year

We integrated a weather API and classified weather conditions into different buckets from terrible weather to ideal conditions.

### Hypothesis
Our hypothesis was that geography, carrier, and time of year were the largest indicators of flight delay outside of realtime data.
* Intuition:
  - Geographic Features are important for two main reasons:
    - Weather: Northern states will have a higher chance of delayed flights due to snow related conditions.
    - Airport Popularity: Some airports are located in highly populated regions or are international airports. These airports are more likely to have delays due to higher traffic.
  - Airline Carriers:
    - Different carriers are different businesses. Each has different policies which effect how they operate.
    - Some carriers are smaller with more dedicated routes. These can be expected to be less likely to have delays.
    - Not all flights carry passengers. Some flights carry shipments of goods, are personal flights, or other things. Each of these could be expected to have different trends.
  - Time of Year:
    - Air Traffic is greatly influenced by seasonality. We can expected influxes in volume around holidays and summer time.
    - Time of year affects weather conditions and therefore delays for different regions differently.

### Methods

SQL, Pandas, and Plotly were used for data exploration
KNN, Random Forrest, Decision Tree, and XGBoost were used for modeling.
SMOTE sampling for data imbalances
GridSearchCV for hyperparameter tuning
Pickle for model saving

### Results



Our model was able to predict any given flight as either delayed or not delayed with an 81% accuracy 7 days in advance using XGBoost while other methods were around 50-60% accuracy.
