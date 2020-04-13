# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:33:53 2020

@author: Marwan
"""
import pandas as pd
import numpy as np
import seaborn
import scipy
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 
import statsmodels.stats.multicomp as multi

df = pd.read_csv("gapminder.csv", low_memory = False, index_col = 0)
maleemployrate = pd.read_csv("maleemployrate_sub.csv", low_memory = False, index_col = 0)

df["incomeperperson"] = df["incomeperperson"].convert_objects(convert_numeric=True)
df["femaleemployrate"] = df["femaleemployrate"].convert_objects(convert_numeric=True)
df["polityscore"] = df["polityscore"].convert_objects(convert_numeric=True)
df['employrate'] = df['employrate'].convert_objects(convert_numeric=True)

maleemployrate['2007'] = maleemployrate['2007'].convert_objects(convert_numeric=True)

df['incomeperperson']=df['incomeperperson'].replace(' ', np.nan)
df['polityscore']=df['polityscore'].replace(' ', np.nan)
df_clean=df.dropna()

#cut into 5 groups
ipp_mean = df_clean['incomeperperson'].mean()
first = df_clean['incomeperperson'].quantile(q=0.2)
second = df_clean['incomeperperson'].quantile(q=0.4)
third  = df_clean['incomeperperson'].quantile(q=0.6)
fourth = df_clean['incomeperperson'].quantile(q=0.8)
fifth = df_clean['incomeperperson'].quantile(q=1)

def income5groups(row):
    if row['incomeperperson'] <= first:
        return "1_Terrible"
    elif row['incomeperperson'] <=second:
        return "2_Bad"
    elif row['incomeperperson'] <= third:
        return "3_Average"
    elif row['incomeperperson'] <= fourth:
        return "4_Decent"
    elif row['incomeperperson'] <= fifth:
        return "5_Great"
        
df_clean['income5groups'] = df_clean.apply(lambda row : income5groups(row),axis=1)
df_clean['income5groups'] = df_clean['income5groups'].astype('category')


sub1=df_clean[(df_clean['income5groups']== '1_Terrible')]
sub2=df_clean[(df_clean['income5groups']== '2_Bad')]
sub3=df_clean[(df_clean['income5groups']== '3_Average')]
sub4=df_clean[(df_clean['income5groups']== '4_Decent')]
sub5=df_clean[(df_clean['income5groups']== '5_Great')]

print (scipy.stats.pearsonr(df_clean['polityscore'], df_clean['femaleemployrate']))


print ('association between polityscore and femaleemployrate for Terrible income countries')
print (scipy.stats.pearsonr(sub1['polityscore'], sub1['femaleemployrate']))
print ('       ')
print ('association between polityscore and femaleemployrate for Bad income countries')
print (scipy.stats.pearsonr(sub2['polityscore'], sub2['femaleemployrate']))
print ('       ')
print ('association between polityscore and femaleemployrate for Average income countries')
print (scipy.stats.pearsonr(sub3['polityscore'], sub3['femaleemployrate']))
print ('       ')
print ('association between polityscore and femaleemployrate for Decent income countries')
print (scipy.stats.pearsonr(sub4['polityscore'], sub4['femaleemployrate']))
print ('       ')
print ('association between polityscore and femaleemployrate for Great income countries')
print (scipy.stats.pearsonr(sub5['polityscore'], sub5['femaleemployrate']))
print ('       ')

#Scatterplot
scat1 = seaborn.regplot(x="polityscore", y="femaleemployrate", data=sub1, label = 'Terrible')
plt.xlabel('polityscore')
plt.ylabel('femaleemployrate')
plt.title('Scatterplot for the Association Between polityscore and femaleemployrate for Terrible income countries')
print (scat1)
scat2 = seaborn.regplot(x="polityscore", y="femaleemployrate", data=sub2, label = 'Bad')
plt.xlabel('polityscore')
plt.ylabel('femaleemployrate')
plt.title('Scatterplot for the Association Between polityscore and femaleemployrate for Bad income countries')
print (scat2)
scat3 = seaborn.regplot(x="polityscore", y="femaleemployrate", data=sub3, label = 'Average')
plt.xlabel('polityscore')
plt.ylabel('femaleemployrate')
plt.title('Scatterplot for the Association Between polityscore and femaleemployrate for Average income countries')
print (scat3)
scat4 = seaborn.regplot(x="polityscore", y="femaleemployrate", data=sub4, label = 'Decent')
plt.xlabel('polityscore')
plt.ylabel('femaleemployrate')
plt.title('Scatterplot for the Association Between polityscore and femaleemployrate for Decent income countries')
print (scat4)
scat5 = seaborn.regplot(x="polityscore", y="femaleemployrate", data=sub5, label = 'Great')
plt.xlabel('polityscore')
plt.ylabel('femaleemployrate')
plt.legend(frameon=True,loc='lower right').get_frame().set_color('white')

plt.title('Scatterplot for the Association Between polityscore and femaleemployrate for income countries')
print (scat5)

