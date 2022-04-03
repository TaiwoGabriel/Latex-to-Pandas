# Pandas data structures
# There are two types of data structures in pandas: Series and DataFrames.
#
# Series: a pandas Series is a one dimensional data structure (“a one dimensional ndarray”)
# that can store values — and for every value it holds a unique index, too.

# DataFrame: a pandas DataFrame is a two (or more) dimensional data structure –
# basically a table with rows and columns. The columns have names and the rows have indexes.

"""
import pandas as pd

df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/zoo.csv'
df = pd.read_csv(df,delimiter=',')
#print(df)

# Data log of a travel blog. This is a log of one day only (if you are a JDS course participant,
# you will get much more of this data set on the last week of the course ;-)).
# I guess the names of the columns are fairly self-explanatory.
df2 = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/pandas_tutorial_read.csv'
article_read = pd.read_csv(df2, delimiter=';', names=['my_datetime', 'event', 'country', 'user_id', 'source', 'topic'])
#print(article_read)

# Does something feel off? Yes, this time we didn’t have a header in our csv file,
# so we have to set it up manually! Add the names parameter to your function!

# Selecting data from a dataframe in pandas
# Printing first five samples
print(article_read.head(5))

# Printing last five samples
print(article_read.tail(5))


# printing five random samples
print(article_read.sample(5))


# Select specific columns of your dataframe
# Print the ‘country’ and the ‘user_id’ columns only.
col = article_read[['country','user_id']]
print(col)

# Note: For the double brackets, the outer bracket frames tell pandas that you want to select columns,
# and the inner brackets are for the list (remember? Python lists go between bracket frames) of the column names.
#By the way, if you change the order of the column names, the order of the returned columns will change, too:
col = article_read[['user_id','country']]
print(col)

# Pandas SERIES
#Sometimes (especially in predictive analytics projects), you want to get Series objects instead of DataFrames.
# You can get a Series using any of these two syntaxes (and selecting only one column):
# Two syntax can be used
col= article_read.user_id
print(col)
print()
# OR
col = article_read['user_id']
print(col)


# Filter for specific values in your dataframe
# Filter for the ‘SEO’ value in the ‘source’ column
col_filter = article_read[article_read.source == 'SEO']
print(col_filter)
# It’s worth it to understand how pandas thinks about data filtering:
#
# STEP 1) First, between the bracket frames it evaluates every line:
# is the article_read.source column’s value 'SEO' or not? The results are boolean values (True or False).
# STEP 2) Then from the article_read table, it prints every row where this value is True and doesn’t
# print any row where it’s False.


# Functions can be used after each other
# It’s very important to understand that pandas’s logic is very linear (compared to SQL, for instance).
# So if you apply a function, you can always apply another one on it. In this case,
# the input of the latter function will always be the output of the previous function.
#
# E.g. combine these two selection methods:
func = article_read.head()[['country','user_id']]
print(func)
print()
# This line first selects the first 5 rows of our data set. And then it takes only
# the ‘country’ and the ‘user_id’ columns.
#
# Could you get the same result with a different chain of functions? Of course you can:
func = article_read[['country','user_id']].head()
print(func)
# In this version, you select the columns first, then take the first five rows.
# The result is the same – the order of the functions (and the execution) is different.


# Select the user_id, the country and the topic columns for the users who are from country_2!
# Print the first five rows only!
col_filter = article_read[article_read.country == 'country_2']
col=col_filter[['user_id','country','topic']]
filtered_col = col.head()
print(filtered_col)

# First you take your original dataframe (article_read), then you filter for the rows where
# the country value is country_2 ([article_read.country == 'country_2']),
# then you take the three columns that were required ([['user_id','topic', 'country']])
# and eventually you take the first five rows only (.head())


col_filter = article_read[article_read.country == 'country_2']
col = col_filter[['user_id','country','topic']]
print(col)


# Data aggregation and Grouping
# Aggregation is the process of turning the values of a dataset (or a subset of it) into one single value
# The operation include min, max, sum, count, etc and grouping

import pandas as pd
import numpy as np

df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/zoo.csv'
zoo = pd.read_csv(df, delimiter=',')

print(zoo)
print()

# Filtering animal column where animal is elephant
col = zoo[zoo.animal == 'elephant']
col_fil = col[['animal','uniq_id','water_need']]
#print(col_fil)

# Counting the number of the animals using a count function count() on the zoo dataframe:
#print(zoo.count())
# the .count() function counts the number of values in each column.
# In the case of the zoo dataset, there were 3 columns, and each of them had 22 values in it.

# Counting the number of values in specific column like water_need column
my_count = zoo[['water_need']].count()
print(my_count) # This print the result in a pandas dataframe

mycount = zoo.water_need.count()
print(mycount) # This print the result as a pandas series


# Sum()
# Sum of all columns
print(zoo.sum())
print()

# you can easily sum the values in the water_need column by typing:
col_sum =zoo[['water_need']].sum()
print(col_sum)


# Min() and Max(): used to fine the smallest and the largest value in a set of values
min_value = zoo.min()
max_value = zoo.max()
print(min_value) # This prints the minimum value in each column
print()
print(max_value) # This prints the maximum value in each column
print()

# Filtering the minimum value in the water_need column
water_min = zoo[['water_need']].min()
print(water_min)
print()

# Filtering the maximum value in the water_need column
water_max = zoo[['water_need']].max()
print(water_max)


# Calculate statistical averages, like mean and median
mean_value = zoo.mean()
median_value = zoo.median()
print(mean_value) # Calculates the mean of every column
print()
print(median_value) # Calculates the median of every column in the data
print()

# Calculate the mean of the water_need column
water_mean = zoo[['water_need']].mean()
print(water_mean)
print()

# calculate the median of the water_need column
water_median = zoo[['water_need']].median()
print(water_median)


# Grouping in pandas
# it’s much more actionable to break this number down – let’s say – by animal types.
# With that, we can compare the species to each other – or we can find outliers.
#
# Here we show how pandas performs “segmentation” (grouping and aggregation) based on the column values

# Pandas .groupby() in action
# Let’s do the above presented grouping and aggregation for real, on our zoo DataFrame!
# We have to fit in a groupby keyword between our zoo variable and our .mean() function:

grp = zoo.groupby('animal').mean()
print(grp)

# Just as before, pandas automatically runs the .mean() calculation for all remaining columns
# (the animal column obviously disappeared, since that was the column we grouped by).
# You can either ignore the uniq_id column, or you can remove it afterwards by using one of these syntaxes:

grp = zoo.groupby('animal').mean()[['water_need']]
print(grp) # Returns a DataFrame


import pandas as pd
import numpy as np

df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/pandas_tutorial_read.csv'
article_read = pd.read_csv(df, delimiter=';', names=['my_datetime', 'event', 'country', 'user_id', 'source', 'topic'])
print(article_read)
print()

# What’s the most frequent source in the article_read dataframe?

freq = article_read.groupby('source').count()
print(freq)
print()

# You can – optionally – remove the unnecessary columns and keep the user_id column only:
freq2 = article_read.groupby('source').count()[['user_id']]
print(freq2)
print()

# For the users of country_2, what was the most frequent topic and source combination?
# Or in other words: which topic, from which source, brought the most views from country_2?
# The result is: the combination of Reddit (source) and Asia (topic), with 139 reads!
comb = article_read[article_read.country == 'country_2'].groupby(['source','topic']).count()
print(comb)


#1. Important Data Formatting Methods (merge, sort, reset_index, fillna)
# Pandas Merge (a.k.a. “joining” dataframes)
# In real life data projects, we usually don’t store all the data in one big data table.
# We store it in a few smaller ones instead
# The point is that it’s quite usual that during your analysis you have to
# pull your data from two or more different tables. The solution for that is called merge.

import pandas as pd
import numpy as np

df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/zoo.csv'
zoo = pd.read_csv(df, delimiter=',')
df2 = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/zoo_eat.csv'
zoo_eat = pd.read_csv(df2, delimiter=';')

print(zoo_eat)
print()
print(zoo)
print()

# Now merge the zoo DataFrame and the zoo_eat DataFrame using merge()
my_merge = zoo.merge(zoo_eat)
#print(my_merge)


# Pandas Merge… But how? Inner, outer, left or right?
# One of the most important questions is how you want to merge these tables
# Inner merge is intersection in set which is by default, outer merge is union in set,
# left merge is zoo only while right merge is zoo_eat only
# When you do an INNER JOIN (that’s the default both in SQL and pandas),
# you merge only those values that are found in both tables. On the other hand,
# when you do the OUTER JOIN, it merges all values, even if you can find some of them in only one of the tables.
mymerge = zoo.merge(zoo_eat, how='outer')
print(mymerge)
print()

# See? Lions came back, the giraffe came back… The only thing is that we have empty (NaN) values in
# those columns where we didn’t get information from the other table.
#
# In my opinion, in this specific case, it would make more sense to keep lions in the table but not the giraffes…
# With that, we could see all the animals in our zoo and we would have three
# food categories: vegetables, meat and NaN (which is basically “no information”).
# Keeping the giraffe line would be misleading and irrelevant since we don’t have any giraffes in our zoo anyway.
# That’s when merging with a how = 'left' parameter becomes handy!
# Again, For doing the merge, pandas needs the key-columns you want to base the merge on
# (in our case it was the animal column in both tables). If you are not so lucky that pandas automatically
# recognizes these key-columns, you have to help it by providing the column names.
# That’s what the left_on and right_on parameters are for!

# Left merge
left_merge = zoo.merge(zoo_eat, how='left', left_on='animal', right_on='animal')
print(left_merge)
print()

# Everything you do need, and nothing you don’t… The how = 'left' parameter brought all the
# values from the left table (zoo) but brought only those values from the right table (zoo_eats)
# that we have in the left one, too
# Notice that the giraffe row has vanished from the table


#2. Fillna
# Remember in the zoo dataset, the problem is that we have NaN values for lions.
# NaN itself can be really distracting, so I usually like to replace it with something more meaningful.
# In some cases, this can be a 0 value, or in other cases a specific string value, but this time, I’ll go with unknown.
# Let’s use the fillna() function, which basically finds and replaces all NaN values in our dataframe with unknown

my_fill = zoo.merge(zoo_eat, how='left').fillna('unknown')
print(my_fill)
print()

# Note: since we know that lions eat meat, we could replace unknown with meat as well
my_fill = zoo.merge(zoo_eat, how='left').fillna('meat')
print(my_fill)


#3. Sorting in pandas
# Quite often, you have to sort by multiple columns, so in general, I recommend using the by keyword for the columns
my_sort = zoo.sort_values(by=['animal', 'water_need'])
print(my_sort)
print()

# Sorting a single column
my_sort = zoo.sort_values(by=['water_need'])
print(my_sort)


# sort_values sorts in ascending order, but obviously, you can change this and do descending order as well:
my_sort = zoo.sort_values(by=['water_need'], ascending=False)
print(my_sort)

#4. Reset_index
# What a mess with all the indexes after that last sorting, right?
# It’s not just that it’s ugly… wrong indexing can mess up your visualizations
# (more about that in my matplotlib tutorials) or even your machine learning models.
#
# The point is: in certain cases, when you have done a transformation on your dataframe,
# you have to re-index the rows. For that, you can use the reset_index() method

my_sort = zoo.sort_values(by=['water_need'],ascending=False).reset_index()
print(my_sort)
print()

# As you can see, our new dataframe kept the old indexes, too. just add the drop = True parameter:
my_sort = zoo.sort_values(['water_need'],ascending=False).reset_index(drop=True)
print(my_sort)


# Test: using the pandas_tutorial_dataset
# TASK #1: What’s the average (mean) revenue between 2018-01-01 and 2018-01-07
# from the users in the article_read dataframe?
# TASK #2: Print the top 3 countries by total revenue between 2018-01-01 and 2018-01-07!
# (Obviously, this concerns the users in the article_read dataframe again.)


import pandas as pd
import numpy as np
df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/pandas_tutorial_read.csv'
article_read = pd.read_csv(df, delimiter=';', names=['my_datetime', 'event', 'country', 'user_id', 'source', 'topic'])
df2 = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/pandas_tutorial_buy_blog.csv'
buy_blog = pd.read_csv(df2, delimiter=';', names=['my_datetime', 'event', 'user_id', 'amount'])

# Task 2
step_1 = article_read.merge(buy_blog, how='left', left_on='user_id', right_on='user_id')
#step_2 = step_1.mean()
print(buy_blog)
step_2 = step_1['amount']
step_3 = step_2.fillna(0)
step_4 = step_3.mean()
print(step_4)

# Note: for ease of understanding, I broke this down into “steps” – but you could also
# bring all these functions into one line.
#
# A short explanation:
#
# (On the screenshot, at the beginning, I included the two extra cells where I import pandas and numpy,
# and where I read the csv files into my Jupyter Notebook.)
# In step_1, I merged the two tables (article_read and blog_buy) based on the user_id columns.
# I kept all the readers from article_read, even if they didn’t buy anything, because 0s should be
# counted in to the average revenue value. And I removed everyone who bought something but wasn’t
# in the article_read dataset (that was fixed in the task). So all in all that led to a left join.
# In step_2, I removed all the unnecessary columns, and kept only amount.
# In step_3, I replaced NaN values with 0s.
# And eventually I did the .mean() calculation.

# Task 2
step_1 = article_read.merge(buy_blog, how = 'left', left_on = 'user_id', right_on = 'user_id')
step_2 = step_1.fillna(0)
step_3 = step_2.groupby('country').sum()
step_4 = step_3.amount
step_5 = step_4.sort_values(ascending = False)
step_5.head(3)

# A short explanation:
#
# At step_1, I used the same merging method that I used in TASK #1.
# At step_2, I filled up all the NaN values with 0s.
# At step_3, I summarized the numerical values by countries.
# At step_4, I took away all columns but amount.
# And at step_5, I sorted the results in descending order, so I can see my top list!
# Finally, I printed the first 3 lines only.


# Plot a Histogram in Python (Using Pandas)
# I have a strong opinion about visualization in Python, which is: it should be useful and not pretty.
#
# Why? Because the fancy data visualization for high-stakes presentations should happen in tools
# that are the best for it: Tableau, Google Data Studio, PowerBI, etc… Creating charts and graphs
# natively in Python should serve only one purpose: to make your data science tasks
# (e.g. prototyping machine learning models) easier and more intuitive

# Histogram
# A histogram shows the number of occurrences of different values in a dataset.
# At first glance, it is very similar to a bar chart.
# Bar chart that shows the frequency of unique values in the dataset
# But when you plot a histogram, there’s one more initial step:
# these unique values will be grouped into ranges. These ranges are called bins or buckets —
# and in Python, the default number of bins is 10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get Data
# Generate two datasets, with 250 data points for female and male.
# And because I fixed the parameter of the random generator (with the np.random.seed() line),
# you’ll get the very same numpy arrays with the very same data points that I have

# Female height
mu = 168 #mean
sigma = 5 #stddev
sample = 250
np.random.seed(0)
height_f = np.random.normal(mu, sigma, sample).astype(int)


# Male height
mu = 176 #mean
sigma = 6 #stddev
sample = 250
np.random.seed(1)
height_m = np.random.normal(mu, sigma, sample).astype(int)

gym = pd.DataFrame({'Female height': height_f, 'Male height': height_f})
print(gym) # We have the heights of female and male gym members in one big 250-row dataframe.

# Plot the Histogram
#gym.hist() # By default, the bins value is 10
#plt.show()

# Chaning Bins and Ranges
# If you want a different amount of bins/buckets than the default 10, you can set that as a parameter. E.g
#gym.hist(bins=20)
#plt.show()

# Plot your histograms on the same chart!
# Sometimes, you want to plot histograms in Python to compare two different columns of your dataframe

gym.plot.hist(bins=20)
#plt.show()

# Note: in this version, you called the .hist() function from .plot.
#
# Anyway, since these histograms are overlapping each other,
# I recommend setting their transparency to 70% by using the alpha() parameter:

gym.plot.hist(bins=20, alpha=0.7)
plt.show()

"""









# Scatter Plot
# two ways to create your scatter plot.
#
# a pandas scatter plot
# and
# a matplotlib scatter plot
# The four steps to creating scatter plots include
# Importing pandas, numpy and matplotlib
# Getting the data
# Preparing the data
# Plotting a scatter plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get your data
#np.random.seed(0)
mu = 170 #mean
sigma = 6 #stddev
sample = 100
height = np.random.normal(mu, sigma, sample)
weight = (height-100) * np.random.uniform(0.75, 1.25, 100)
# This is a random generator, by the way, that generates 100 height and 100 weight values — in numpy array format.
# By using the np.random.seed(0) line, we also made sure you’ll be able to work with the exact same data points

# Prepare the data
# Again, preparing, cleaning and formatting the data is a painful and
# time consuming process in real-life data science projects. But in this tutorial,
# we are lucky, everything is prepared – the data is clean – so you can push your height and weight
# data sets directly into a pandas dataframe.

gym = pd.DataFrame({'height': height, 'weight': weight})
#print(gym)

# Putting the data on a scatterplot
#1. Using pandas
gym.plot.scatter(x='weight', y='height')
#plt.show()

#2. using matplotlib()
x = gym.weight
y = gym.height
plt.scatter(x,y)
plt.show()
# This solution is a bit more elegant. But from a technical standpoint — and for results —
# both solutions are equally great.
