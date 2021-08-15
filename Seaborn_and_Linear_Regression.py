"""Film budgets vs box office revenue analysis

This 'script' analyzes a film bugdgets vs box office revenue dataset to answer questions such as:
    Is there a relationship between movie budgets and box office revenue?
    What is the trend of the quantity of movie releases over time?
    How much money would a X budget film generate according to our linear regression?
A linear regression of the data is performed using sklearn. The results of the analysis are visualized using Seaborn.

This script requires that 'pandas', 'Matplotlib', 'Seaborn', and 'scikit-learn' be installed within the Python
environment you are running this script in.

"""

#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Do higher film budgets lead to more box office revenue? Let's find out if there's a relationship using the movie budgets and financial performance data that I've scraped from [the-numbers.com](https://www.the-numbers.com/movie/budgets) on **May 1st, 2018**. 
# 
# <img src=https://i.imgur.com/kq7hrEh.png>

# # Import Statements

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# # Notebook Presentation

# In[3]:


pd.options.display.float_format = '{:,.2f}'.format

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# # Read the Data

# In[4]:


data = pd.read_csv('day77-data/cost_revenue_dirty.csv')


# # Explore and Clean the Data

# **Challenge**: Answer these questions about the dataset:
# 1. How many rows and columns does the dataset contain?
# 2. Are there any NaN values present?
# 3. Are there any duplicate rows?
# 4. What are the data types of the columns?

# In[5]:


data.shape


# In[6]:


data.isna().any()


# In[7]:


data.duplicated().any()


# In[8]:


data.dtypes


# ### Data Type Conversions

# **Challenge**: Convert the `USD_Production_Budget`, `USD_Worldwide_Gross`, and `USD_Domestic_Gross` columns to a numeric format by removing `$` signs and `,`. 
# <br>
# <br>
# Note that *domestic* in this context refers to the United States.

# In[9]:


data.head()


# In[10]:


columns = ['USD_Production_Budget', 'USD_Worldwide_Gross', 'USD_Domestic_Gross' ]
remove_characters = ['$', ',']

for column in columns:
    for character in remove_characters:
        data[column]=data[column].str.replace(character,'')
    data[column] = data[column].astype('float')


# In[11]:


data.dtypes


# **Challenge**: Convert the `Release_Date` column to a Pandas Datetime type. 

# In[12]:


data['Release_Date'] = pd.to_datetime(data['Release_Date'])


# ### Descriptive Statistics

# **Challenge**: 
# 
# 1. What is the average production budget of the films in the data set?
# 2. What is the average worldwide gross revenue of films?
# 3. What were the minimums for worldwide and domestic revenue?
# 4. Are the bottom 25% of films actually profitable or do they lose money?
# 5. What are the highest production budget and highest worldwide gross revenue of any film?
# 6. How much revenue did the lowest and highest budget films make?

# In[13]:


data.head()


# <b>What is the mean production budget of the films in the data set?</b>

# In[14]:


data['USD_Production_Budget'].mean()


# <b>What is the average worldwide gross revenue of films?</b>

# In[15]:


data['USD_Worldwide_Gross'].mean()


# <b>What were the minimums for worldwide and domestic revenue?</b>

# In[16]:


data[['USD_Worldwide_Gross', 'USD_Domestic_Gross']].mean()


# <b>Are the bottom 25% of films actually profitable or do they lose money?</b>

# In[17]:


data.sort_values('USD_Production_Budget', ascending=True)[:int(len(data)/4)].head()


# <b>What are the highest production budget and highest worldwide gross revenue of any film?</b>

# In[18]:


data[['USD_Production_Budget','USD_Worldwide_Gross']].max()


# <b>How much revenue did the lowest and highest budget films make?</b>

# In[19]:


data.loc[[data['USD_Production_Budget'].idxmin(),data['USD_Production_Budget'].idxmax()]]


# # Investigating the Zero Revenue Films

# In[20]:


data.head()


# **Challenge** How many films grossed $0 domestically (i.e., in the United States)? What were the highest budget films that grossed nothing?

# In[21]:


len(data[data['USD_Domestic_Gross']==0])


# In[22]:


data[data['USD_Domestic_Gross']==0].sort_values('USD_Production_Budget', ascending=False).head()


# **Challenge**: How many films grossed $0 worldwide? What are the highest budget films that had no revenue internationally?

# In[23]:


len(data[data['USD_Worldwide_Gross']==0])


# In[24]:


data[data['USD_Worldwide_Gross']==0].sort_values('USD_Production_Budget', ascending=False).head()


# ### Filtering on Multiple Conditions

# In[25]:


mask1 = data['USD_Worldwide_Gross'] > 0
mask2 = data['USD_Domestic_Gross'] == 0
data[mask1 & mask2].head()


# **Challenge**: Use the [`.query()` function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) to accomplish the same thing. Create a subset for international releases that had some worldwide gross revenue, but made zero revenue in the United States. 
# 
# Hint: This time you'll have to use the `and` keyword.

# In[26]:


data.query('USD_Worldwide_Gross > 0 and USD_Domestic_Gross ==0').head()


# ### Unreleased Films
# 
# **Challenge**:
# * Identify which films were not released yet as of the time of data collection (May 1st, 2018).
# * How many films are included in the dataset that have not yet had a chance to be screened in the box office? 
# * Create another DataFrame called data_clean that does not include these films. 

# In[27]:


# Date of Data Collection
scrape_date = pd.Timestamp('2018-5-1')


# In[28]:


data[data['Release_Date']>scrape_date]


# In[29]:


data_clean = data.drop(labels=data[data['USD_Production_Budget'] > data['USD_Worldwide_Gross']].index)
data_clean.head()


# ### Films that Lost Money
# 
# **Challenge**: 
# What is the percentage of films where the production costs exceeded the worldwide gross revenue? 

# In[30]:


len(data[data['USD_Production_Budget'] > data['USD_Worldwide_Gross']]) / len(data) * 100


# # Seaborn for Data Viz: Bubble Charts

# In[31]:


plt.figure(figsize=(8,4), dpi=200)
with sns.axes_style('darkgrid'):
    ax = sns.scatterplot(
        data=data_clean,
        x='USD_Production_Budget',
        y='USD_Worldwide_Gross',
        hue='USD_Worldwide_Gross',
        size='USD_Worldwide_Gross'
    )
    ax.set(
        xlim=(0, 450000000),
        ylim=(0, 3000000000),
        xlabel='Budget in $100 Millions',
        ylabel='Revenue in $ billions'
    )
    plt.show()


# ### Plotting Movie Releases over Time
# 
# **Challenge**: Try to create the following Bubble Chart:
# 
# <img src=https://i.imgur.com/8fUn9T6.png>
# 
# 

# In[32]:


plt.figure(figsize=(14,8), dpi=200)
with sns.axes_style('darkgrid'):
    ax = sns.scatterplot(
        data=data_clean,
        x='Release_Date',
        y='USD_Production_Budget',
        hue='USD_Worldwide_Gross',
        size='USD_Worldwide_Gross'
    )
    ax.set(
        xlim=(data_clean['Release_Date'].min(), data_clean['Release_Date'].max()),
        ylim=(0, 450000000),
        xlabel='Release Year',
        ylabel='Budget in $100 Millions',
    )
    plt.show()


# # Converting Years to Decades Trick
# 
# **Challenge**: Create a column in `data_clean` that has the decade of the release. 
# 
# <img src=https://i.imgur.com/0VEfagw.png width=650> 
# 
# Here's how: 
# 1. Create a [`DatetimeIndex` object](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html) from the Release_Date column. 
# 2. Grab all the years from the `DatetimeIndex` object using the `.year` property.
# <img src=https://i.imgur.com/5m06Ach.png width=650>
# 3. Use floor division `//` to convert the year data to the decades of the films.
# 4. Add the decades as a `Decade` column to the `data_clean` DataFrame.

# In[33]:


datetimeindex = pd.DatetimeIndex(data_clean['Release_Date'])
yearsindex = datetimeindex.year
data_clean['Decade'] = yearsindex // 10*10


# In[34]:


data_clean.head()


# ### Separate the "old" (before 1969) and "New" (1970s onwards) Films
# 
# **Challenge**: Create two new DataFrames: `old_films` and `new_films`
# * `old_films` should include all the films before 1969 (up to and including 1969)
# * `new_films` should include all the films from 1970 onwards
# * How many films were released prior to 1970?
# * What was the most expensive film made prior to 1970?

# In[46]:


old_films = data_clean[data_clean['Decade'] <= 1960]
new_films = data_clean[data_clean['Decade'] > 1970]
print(f'Here is the number of films released prior to 1970: {len(old_films)}')
print(f'Here is the number of films released after 1970: {len(new_films)}')


# In[36]:


old_films.loc[old_films['USD_Production_Budget'].idxmax()]


# # Seaborn Regression Plots

# In[37]:


sns.regplot(data=old_films,
           x='USD_Production_Budget',
           y='USD_Worldwide_Gross')


# In[38]:


plt.figure(figsize=(8,4), dpi=200)
with sns.axes_style('darkgrid'):
    ax=sns.regplot(
        data=old_films,
        x='USD_Production_Budget',
        y='USD_Worldwide_Gross',
        scatter_kws={'alpha': 0.4},
        line_kws = {'color':'black'}
    )
    ax.set(
        xlim=(0, 42500000)
    )


# **Challenge**: Use Seaborn's `.regplot()` to show the scatter plot and linear regression line against the `new_films`. 
# <br>
# <br>
# Style the chart
# 
# * Put the chart on a `'darkgrid'`.
# * Set limits on the axes so that they don't show negative values.
# * Label the axes on the plot "Revenue in \$ billions" and "Budget in \$ millions".
# * Provide HEX colour codes for the plot and the regression line. Make the dots dark blue (#2f4b7c) and the line orange (#ff7c43).
# 
# Interpret the chart
# 
# * Do our data points for the new films align better or worse with the linear regression than for our older films?
# * Roughly how much would a film with a budget of $150 million make according to the regression line?

# In[39]:


with sns.axes_style('darkgrid'):
    plt.figure(figsize=(8,4), dpi=200)
    plt.title('Budget vs Revenue')
    ax = sns.regplot(
        data=new_films,
        x=new_films['USD_Production_Budget'],
        y=new_films['USD_Worldwide_Gross'],
        scatter_kws={'color':'#2f4b7c'},
        line_kws={'color': '#ff7c43'}
    )
    ax.set(
        xlim=(0, 430000000),
        ylim=(0, 2850000000),
        xlabel=('Budget in $100 million'),
        ylabel=('Revenue in $ billion')
    )
    plt.show()


# # Run Your Own Regression with scikit-learn
# 
# $$ REV \hat ENUE = \theta _0 + \theta _1 BUDGET$$

# In[40]:


regression = LinearRegression()


# **Challenge**: Run a linear regression for the `new_films`. Calculate the intercept, slope and r-squared. How much of the variance in movie revenue does the linear model explain in this case?

# In[47]:


X = pd.DataFrame(new_films, columns=['USD_Production_Budget'])
y = pd.DataFrame(new_films, columns=['USD_Worldwide_Gross'])


# In[48]:


regression.fit(X, y)


# In[49]:


regression.intercept_


# In[50]:


regression.coef_


# In[51]:


regression.score(X, y)


# **Challenge**: Run a linear regression for the `old_films`. Calculate the intercept, slope and r-squared. How much of the variance in movie revenue does the linear model explain in this case?

# In[57]:


X2 = pd.DataFrame(old_films, columns=['USD_Production_Budget'])
y2= pd.DataFrame(old_films, columns=['USD_Worldwide_Gross'])


# In[63]:


regression2 = LinearRegression()


# In[64]:


regression2.fit(X2, y2)


# In[65]:


regression2.intercept_


# In[66]:


regression2.coef_


# In[67]:


regression2.score(X2, y2)


# In[77]:


budget = 350000000
revenue = regression2.intercept_[0] + regression2.coef_[0,0]*budget
revenue = round(revenue, 2)
print(f'The estimated revenue for $350MM film is around ${revenue:.11}.')


# In[81]:





# # Use Your Model to Make a Prediction
# 
# We just estimated the slope and intercept! Remember that our Linear Model has the following form:
# 
# $$ REV \hat ENUE = \theta _0 + \theta _1 BUDGET$$
# 
# **Challenge**:  How much global revenue does our model estimate for a film with a budget of $350 million? 

# In[ ]:





# In[ ]:




