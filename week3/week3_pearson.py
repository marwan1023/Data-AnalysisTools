# Course: Data Analysis Tools
# Week3
#
import pandas as pd
import numpy as np
import seaborn
import scipy
import matplotlib.pyplot as plt

df = pd.read_csv("gapminder.csv", low_memory = False, index_col = 0)
maleemployrate = pd.read_csv("maleemployrate_sub.csv", low_memory = False, index_col = 0)

df["incomeperperson"] = df["incomeperperson"].convert_objects(convert_numeric=True)
df["femaleemployrate"] = df["femaleemployrate"].convert_objects(convert_numeric=True)
df["polityscore"] = df["polityscore"].convert_objects(convert_numeric=True)
df['employrate'] = df['employrate'].convert_objects(convert_numeric=True)
df['internetuserate'] = df['internetuserate'].convert_objects(convert_numeric=True)
df['urbanrate'] = df['urbanrate'].convert_objects(convert_numeric=True)

maleemployrate['2007'] = maleemployrate['2007'].convert_objects(convert_numeric=True)

df['incomeperperson']=df['incomeperperson'].replace(' ', np.nan)
df['polityscore']=df['polityscore'].replace(' ', np.nan)
df['internetuserate']=df['internetuserate'].replace(' ', np.nan)
df['urbanrate']=df['urbanrate'].replace(' ', np.nan)

# Scatterplot 
scat1 = seaborn.regplot(x="polityscore", y="incomeperperson", fit_reg=True, data=df)
plt.xlabel('polity score')
plt.ylabel('income per person')
plt.title('Scatterplot for the Association Between polity score and income per person')
plt.show()

scat2 = seaborn.regplot(x="polityscore", y="femaleemployrate", fit_reg=True, data=df)
plt.xlabel('polity score')
plt.ylabel('female employ rate')
plt.title('Scatterplot for the Association Between female employ rate and polity score')
plt.show()

scat3 = seaborn.regplot(x="internetuserate", y="urbanrate", fit_reg=True, data=df)
plt.xlabel('internetuserate')
plt.ylabel('urbanrate ')
plt.title('Scatterplot for the Association Between internetuserate and urbanrate')
plt.show()

data_clean = df.dropna()
print ('association between polity score and income per person')
print (scipy.stats.pearsonr(data_clean['polityscore'], data_clean['incomeperperson']))

print ('association Between female employ rate and polity score')
print (scipy.stats.pearsonr(data_clean['polityscore'], data_clean['femaleemployrate']))

print ('association Between internetuserate and urbanrate')
print (scipy.stats.pearsonr(data_clean['internetuserate'], data_clean['urbanrate']))