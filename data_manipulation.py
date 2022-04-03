# Pandas DataFrames
# Pandas is a high-level data manipulation tool developed by Wes McKinney. It is built on the Numpy package
# and its key data structure is called the DataFrame. DataFrames allow you to store and manipulate tabular
# data in rows of observations and columns of variables.
#
# There are several ways to create a DataFrame. One way way is to use a dictionary amd the other way is create
# a DataFrame is by importing a csv file using Pandas

import numpy as np
import pandas as pd

#This code below is used to specify the location of the dataset in the system
df = 'C:/Users/Omomule Taiwo G/Desktop/Datasets/wine.csv'

#This code below is used to read in the dataset. Note that the datafile (wine.data) does not have attributes
#written in it, the attributes are in another file, so the "name" parameter is used to specify
#the attributes.
df = pd.read_csv(df, delimiter=',', names=['cultivator','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                                              'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                                              'Proanthocyanins', 'Color intensity', 'Hue',
                                              'OD280/OD315 of diluted wines', 'Proline'])

print(df)


#This code is used to drop or delete a column from a dataset
new_data = df.drop(columns='Alcohol')
print(new_data)


#This code below shows the number of rows and column in the data
print('The shape of the data is',df.shape)


#describing the statistical analysis of the data
print(df.describe())


#selecting first five rows in the dataset
print(df.head(5))


#selecting the last five lines in the dataset
print(df.tail(5))


#selecting five random samples
print(df.sample(5))


#printing columns of the data
# NOTE: Two brackets after the "data" in the print function are necessary. The outer bracket frames
# tell pandas that you want to select columns, and the inner brackets are
# for the list (remember? Python lists go between bracket frames) of the column names.
print(df.head(10)[['Proline','cultivator', 'Ash']])
print(df.head(8)[['Hue', 'Alcohol']])


#filtering all the data for class 3
data = df['cultivator'] == 3
print(data)


#counting the numbers of samples in each row of the dataset
print(df.count())


#counting the numbers of values in a column
print('Flavanoids: ', df['Flavanoids'].count())


#Calculating the sum of values in each column of the dataset
print(df.sum())


#summing the values in a column
print('Alcohol sum:', df['Alcohol'].sum)


#getting the mimimum and maximum value in a column
print('Maximum value of cultivator: ', df['cultivator'].max())
print('Minimum value of cultivator ', df['cultivator'].min())
print()

# Printing the maximum and minimum values in of the data
print('Minimum value of each column in the dataset: \n')
print(df.min())
print('Maximum value of each column in the dataset: \n')
print(df.max())


# mean and median of the data
print('Mean value of each column in the dataset: \n')
print(df.mean())

print('The mean value of the Alcohol column in the dataset: \n')
print(df['Alcohol'].mean())

print('Median value of each column of the dataset: ',)
print(df.median())

print('Median Value of the Alcohol column in the dataset: \n')
print(df['Alcohol'].median)


#grouping of the cultivators by their mean for each feature
print(df.groupby('cultivator').mean())

#We can also groupby using the code below
#print(data.groupby(by='cultivator').mean())


#grouping of the cultivators by their size. This shows the number of samples that belong to the
#first cultivator, the second cultivator and the third cultivator
size_group = df.groupby('cultivator').size()
print(size_group)


#printing first 5 samples from each cultivator (class)
print(df.groupby('cultivator').head(10))


#printing last 5 samples from each cultivator (class)
print(df.groupby('cultivator').tail(10))


import pandas as pd
# Create a simple dataset using list inside dictionary
data = {'Name':['John','Anna', 'Peter', 'Linda'],
        'Location': ['New York','Paris','Berlin','London'],
        'Age':[24,13,53,33]}

df2 = pd.DataFrame(data)
print(df2)



import pandas as pd
data = {'Country':['Brazil','Russia','India','China','South Africa'],
        'Capital':['Brasilla','Moscow','New Dehli','Beijing','Pretoria'],
        'Population':[200.4, 143.5, 1252, 1357, 52.98],
        'Area':[8.516,17.10,3.286,9.597,1.221]}
df3 = pd.DataFrame(data)
#print(df3)

# Printing Countries with age population greater than 150
print(df3[df3.Population > 150])



