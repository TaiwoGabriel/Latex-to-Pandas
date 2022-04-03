# Code to transform pandas table to latex table

import pandas as pd

#------------------------------------------------------------------------------------------------------
#print('----------------------------------------------------------------10-90')
data = "C:/Users/Omomule Taiwo G/Desktop/CHPC/Latex/red_wine/red_wine10.csv"
features = ['unnamed0', 'unnamed1', 'unnamed2','unnamed3']

df = pd.read_csv(data, delim_whitespace=True, names=features)
df['unnamed0'] = df['unnamed0'].str.strip('>')
df = df.drop(['unnamed2','unnamed3'], axis=1)
#print(df, '\n')
test_df = df.head(13)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.rename(columns=test_df.iloc[0]).drop(test_df.index[0])
#print(test_df, '\n')
train_df = df.tail(13)
train_df = train_df.drop('unnamed0', axis=1)
train_df.reset_index(drop=True, inplace=True)
#train_df = train_df.rename(columns=train_df.iloc[0]).drop(train_df.index[0])
train_df = train_df.rename(columns={'unnamed1': 'Training_Accuracy',})
#print(train_df, '\n')

new_df = pd.concat([test_df, train_df], axis=1)
new_df = new_df.round(3)

new_df2 = new_df.rename(columns={'unnamed0': 'Ensemble', 'unnamed1': 'Testing_Accuracy'})

new_df2['Testing_Error'] = 1 - new_df2['Testing_Accuracy']
new_df2['Training_Error'] = 1 - new_df2['Training_Accuracy']
new_df2['GF'] = new_df2['Testing_Error'] / new_df2['Training_Error']
new_df2['GF'] = new_df2['GF'].round(3)
#print(new_df2.to_string(), '\n')

NB1 = new_df2.iloc[0]
#print(NB1)



#-------------------------------------------------------------------------------------------------------------
#print('----------------------------------------------------------------15-85')
data1 = "C:/Users/Omomule Taiwo G/Desktop/CHPC/Latex/red_wine/red_wine15.csv"
features1 = ['unnamed0', 'unnamed1', 'unnamed2','unnamed3']

df = pd.read_csv(data1, delim_whitespace=True, names=features1)
df['unnamed0'] = df['unnamed0'].str.strip('>')
df = df.drop(['unnamed2','unnamed3'], axis=1)
#print(df, '\n')
test_df = df.head(13)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.rename(columns=test_df.iloc[0]).drop(test_df.index[0])
#print(test_df, '\n')
train_df = df.tail(13)
train_df = train_df.drop('unnamed0', axis=1)
train_df.reset_index(drop=True, inplace=True)
#train_df = train_df.rename(columns=train_df.iloc[0]).drop(train_df.index[0])
train_df = train_df.rename(columns={'unnamed1': 'Training_Accuracy',})
#print(train_df, '\n')

new_df = pd.concat([test_df, train_df], axis=1)
new_df = new_df.round(3)

new_df3 = new_df.rename(columns={'unnamed0': 'Ensemble', 'unnamed1': 'Testing_Accuracy'})

new_df3['Testing_Error'] = 1 - new_df3['Testing_Accuracy']
new_df3['Training_Error'] = 1 - new_df3['Training_Accuracy']
new_df3['GF'] = new_df3['Testing_Error'] / new_df3['Training_Error']
new_df3['GF'] = new_df3['GF'].round(3)
#print(new_df3.to_string(), '\n')

NB2 = new_df3.iloc[0]




#-------------------------------------------------------------------------------------------------------------
#print('----------------------------------------------------------------20-80')
data2 = "C:/Users/Omomule Taiwo G/Desktop/CHPC/Latex/red_wine/red_wine20.csv"
features2 = ['unnamed0', 'unnamed1', 'unnamed2','unnamed3']

df = pd.read_csv(data2, delim_whitespace=True, names=features2)
df['unnamed0'] = df['unnamed0'].str.strip('>')
df = df.drop(['unnamed2','unnamed3'], axis=1)
#print(df, '\n')
test_df = df.head(13)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.rename(columns=test_df.iloc[0]).drop(test_df.index[0])
#print(test_df, '\n')
train_df = df.tail(13)
train_df = train_df.drop('unnamed0', axis=1)
train_df.reset_index(drop=True, inplace=True)
#train_df = train_df.rename(columns=train_df.iloc[0]).drop(train_df.index[0])
train_df = train_df.rename(columns={'unnamed1': 'Training_Accuracy',})
#print(train_df, '\n')

new_df = pd.concat([test_df, train_df], axis=1)
new_df = new_df.round(3)

new_df4 = new_df.rename(columns={'unnamed0': 'Ensemble', 'unnamed1': 'Testing_Accuracy'})

new_df4['Testing_Error'] = 1 - new_df4['Testing_Accuracy']
new_df4['Training_Error'] = 1 - new_df4['Training_Accuracy']
new_df4['GF'] = new_df4['Testing_Error'] / new_df4['Training_Error']
new_df4['GF'] = new_df4['GF'].round(3)
#print(new_df4.to_string(), '\n')

NB3 = new_df4.iloc[0]


#-------------------------------------------------------------------------------------------------------------
#print('----------------------------------------------------------------25-75')
data3 = "C:/Users/Omomule Taiwo G/Desktop/CHPC/Latex/red_wine/red_wine25.csv"
features3 = ['unnamed0', 'unnamed1', 'unnamed2','unnamed3']

df = pd.read_csv(data3, delim_whitespace=True, names=features3)
df['unnamed0'] = df['unnamed0'].str.strip('>')
df = df.drop(['unnamed2','unnamed3'], axis=1)
#print(df, '\n')
test_df = df.head(13)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.rename(columns=test_df.iloc[0]).drop(test_df.index[0])
#print(test_df, '\n')
train_df = df.tail(13)
train_df = train_df.drop('unnamed0', axis=1)
train_df.reset_index(drop=True, inplace=True)
#train_df = train_df.rename(columns=train_df.iloc[0]).drop(train_df.index[0])
train_df = train_df.rename(columns={'unnamed1': 'Training_Accuracy',})
#print(train_df, '\n')

new_df = pd.concat([test_df, train_df], axis=1)
new_df = new_df.round(3)

new_df5 = new_df.rename(columns={'unnamed0': 'Ensemble', 'unnamed1': 'Testing_Accuracy'})

new_df5['Testing_Error'] = 1 - new_df5['Testing_Accuracy']
new_df5['Training_Error'] = 1 - new_df5['Training_Accuracy']
new_df5['GF'] = new_df5['Testing_Error'] / new_df5['Training_Error']
new_df5['GF'] = new_df5['GF'].round(3)
#print(new_df5.to_string(), '\n')

NB4 = new_df5.iloc[0]


#-------------------------------------------------------------------------------------------------------------
#print('----------------------------------------------------------------30-70')
data4 = "C:/Users/Omomule Taiwo G/Desktop/CHPC/Latex/red_wine/red_wine30.csv"
features4 = ['unnamed0', 'unnamed1', 'unnamed2','unnamed3']

df = pd.read_csv(data4, delim_whitespace=True, names=features4)
df['unnamed0'] = df['unnamed0'].str.strip('>')
df = df.drop(['unnamed2','unnamed3'], axis=1)
#print(df, '\n')
test_df = df.head(13)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.rename(columns=test_df.iloc[0]).drop(test_df.index[0])
#print(test_df, '\n')
train_df = df.tail(13)
train_df = train_df.drop('unnamed0', axis=1)
train_df.reset_index(drop=True, inplace=True)
#train_df = train_df.rename(columns=train_df.iloc[0]).drop(train_df.index[0])
train_df = train_df.rename(columns={'unnamed1': 'Training_Accuracy',})
#print(train_df, '\n')

new_df = pd.concat([test_df, train_df], axis=1)
new_df = new_df.round(3)

new_df6 = new_df.rename(columns={'unnamed0': 'Ensemble', 'unnamed1': 'Testing_Accuracy'})

new_df6['Testing_Error'] = 1 - new_df6['Testing_Accuracy']
new_df6['Training_Error'] = 1 - new_df6['Training_Accuracy']
new_df6['GF'] = new_df6['Testing_Error'] / new_df6['Training_Error']
new_df6['GF'] = new_df6['GF'].round(3)
#print(new_df6.to_string(), '\n')

NB5 = new_df6.iloc[0]


#-------------------------------------------------------------------------------------------------------------
#print('----------------------------------------------------------------35-65')
data5 = "C:/Users/Omomule Taiwo G/Desktop/CHPC/Latex/red_wine/red_wine35.csv"
features5 = ['unnamed0', 'unnamed1', 'unnamed2','unnamed3']

df = pd.read_csv(data5, delim_whitespace=True, names=features5)
df['unnamed0'] = df['unnamed0'].str.strip('>')
df = df.drop(['unnamed2','unnamed3'], axis=1)
#print(df, '\n')
test_df = df.head(13)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.rename(columns=test_df.iloc[0]).drop(test_df.index[0])
#print(test_df, '\n')
train_df = df.tail(13)
train_df = train_df.drop('unnamed0', axis=1)
train_df.reset_index(drop=True, inplace=True)
#train_df = train_df.rename(columns=train_df.iloc[0]).drop(train_df.index[0])
train_df = train_df.rename(columns={'unnamed1': 'Training_Accuracy',})
#print(train_df, '\n')

new_df = pd.concat([test_df, train_df], axis=1)
new_df = new_df.round(3)

new_df7 = new_df.rename(columns={'unnamed0': 'Ensemble', 'unnamed1': 'Testing_Accuracy'})

new_df7['Testing_Error'] = 1 - new_df7['Testing_Accuracy']
new_df7['Training_Error'] = 1 - new_df7['Training_Accuracy']
new_df7['GF'] = new_df7['Testing_Error'] / new_df7['Training_Error']
new_df7['GF'] = new_df7['GF'].round(3)
#print(new_df7.to_string(), '\n')
NB6 = new_df7.iloc[0]



#-------------------------------------------------------------------------------------------------------------
#print('----------------------------------------------------------------40-60')
data6 = "C:/Users/Omomule Taiwo G/Desktop/CHPC/Latex/red_wine/red_wine40.csv"
features6 = ['unnamed0', 'unnamed1', 'unnamed2','unnamed3']

df = pd.read_csv(data6, delim_whitespace=True, names=features6)
df['unnamed0'] = df['unnamed0'].str.strip('>')
df = df.drop(['unnamed2','unnamed3'], axis=1)
#print(df, '\n')
test_df = df.head(13)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.rename(columns=test_df.iloc[0]).drop(test_df.index[0])
#print(test_df, '\n')
train_df = df.tail(13)
train_df = train_df.drop('unnamed0', axis=1)
train_df.reset_index(drop=True, inplace=True)
#train_df = train_df.rename(columns=train_df.iloc[0]).drop(train_df.index[0])
train_df = train_df.rename(columns={'unnamed1': 'Training_Accuracy',})
#print(train_df, '\n')

new_df = pd.concat([test_df, train_df], axis=1)
new_df = new_df.round(3)

new_df8 = new_df.rename(columns={'unnamed0': 'Ensemble', 'unnamed1': 'Testing_Accuracy'})

new_df8['Testing_Error'] = 1 - new_df8['Testing_Accuracy']
new_df8['Training_Error'] = 1 - new_df8['Training_Accuracy']
new_df8['GF'] = new_df8['Testing_Error'] / new_df8['Training_Error']
new_df8['GF'] = new_df8['GF'].round(3)
#print(new_df8.to_string(), '\n')

NB7 = new_df8.iloc[0]


#-------------------------------------------------------------------------------------------------------------
#print('----------------------------------------------------------------45-55')
data7 = "C:/Users/Omomule Taiwo G/Desktop/CHPC/Latex/red_wine/red_wine45.csv"
features7 = ['unnamed0', 'unnamed1', 'unnamed2','unnamed3']

df = pd.read_csv(data7, delim_whitespace=True, names=features7)
df['unnamed0'] = df['unnamed0'].str.strip('>')
df = df.drop(['unnamed2','unnamed3'], axis=1)
#print(df, '\n')
test_df = df.head(13)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.rename(columns=test_df.iloc[0]).drop(test_df.index[0])
#print(test_df, '\n')
train_df = df.tail(13)
train_df = train_df.drop('unnamed0', axis=1)
train_df.reset_index(drop=True, inplace=True)
#train_df = train_df.rename(columns=train_df.iloc[0]).drop(train_df.index[0])
train_df = train_df.rename(columns={'unnamed1': 'Training_Accuracy',})
#print(train_df, '\n')

new_df = pd.concat([test_df, train_df], axis=1)
new_df = new_df.round(3)

new_df9 = new_df.rename(columns={'unnamed0': 'Ensemble', 'unnamed1': 'Testing_Accuracy'})

new_df9['Testing_Error'] = 1 - new_df9['Testing_Accuracy']
new_df9['Training_Error'] = 1 - new_df9['Training_Accuracy']
new_df9['GF'] = new_df9['Testing_Error'] / new_df9['Training_Error']
new_df9['GF'] = new_df9['GF'].round(3)
#print(new_df9.to_string(), '\n')

NB8 = new_df9.iloc[0]


#-------------------------------------------------------------------------------------------------------------
#print('----------------------------------------------------------------50-50')
data8 = "C:/Users/Omomule Taiwo G/Desktop/CHPC/Latex/red_wine/red_wine50.csv"
features8 = ['unnamed0', 'unnamed1', 'unnamed2','unnamed3']

df = pd.read_csv(data8, delim_whitespace=True, names=features8)
df['unnamed0'] = df['unnamed0'].str.strip('>')
df = df.drop(['unnamed2','unnamed3'], axis=1)
#print(df, '\n')
test_df = df.head(13)
test_df.reset_index(drop=True, inplace=True)
#test_df = test_df.rename(columns=test_df.iloc[0]).drop(test_df.index[0])
#print(test_df, '\n')
train_df = df.tail(13)
train_df = train_df.drop('unnamed0', axis=1)
train_df.reset_index(drop=True, inplace=True)
#train_df = train_df.rename(columns=train_df.iloc[0]).drop(train_df.index[0])
train_df = train_df.rename(columns={'unnamed1': 'Training_Accuracy',})
#print(train_df, '\n')

new_df = pd.concat([test_df, train_df], axis=1)
new_df = new_df.round(3)

new_df10 = new_df.rename(columns={'unnamed0': 'Ensemble', 'unnamed1': 'Testing_Accuracy'})

new_df10['Testing_Error'] = 1 - new_df10['Testing_Accuracy']
new_df10['Training_Error'] = 1 - new_df10['Training_Accuracy']
new_df10['GF'] = new_df10['Testing_Error'] / new_df10['Training_Error']
new_df10['GF'] = new_df10['GF'].round(3)
#print(new_df10.to_string(), '\n')


#----------------------------------------------------------------------------------------------------
# Printing the performance of each ensemble in the skewed class distributions
NB9 = new_df10.iloc[0]
NB_df = pd.concat([NB1, NB2, NB3, NB4, NB5, NB6, NB7, NB8, NB9], axis=1)
NB_df = NB_df.drop('Ensemble', axis=0)
NB_df.columns = ['10-90%', '15-85%', '20-80%', '25-75%', '30-70%', '35-65%', '40-60%', '45-55%', '50-50%']
print('NBE \n',NB_df.to_string())
