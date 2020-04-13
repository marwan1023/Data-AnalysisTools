# Course: Data Analysis Tools
# Week2
#Editor: Kuo-Lin Hsueh
import pandas as pd
import numpy as np
import seaborn
import scipy.stats
import matplotlib.pyplot as plt

df = pd.read_csv("gapminder.csv", low_memory = False, index_col = 0)
maleemployrate = pd.read_csv("maleemployrate_sub.csv", low_memory = False, index_col = 0)

df["incomeperperson"] = df["incomeperperson"].convert_objects(convert_numeric=True)
df["femaleemployrate"] = df["femaleemployrate"].convert_objects(convert_numeric=True)
df["polityscore"] = df["polityscore"].convert_objects(convert_numeric=True)
df['employrate'] = df['employrate'].convert_objects(convert_numeric=True)

maleemployrate['2007'] = maleemployrate['2007'].convert_objects(convert_numeric=True)


#Concatenate df , df2
df3 =  pd.concat([df, maleemployrate], axis=1, join_axes=[df.index])
df3.rename(columns= {'2007':'maleemployrate'}, inplace=True) #rename column

## Group by mean
ipp_mean = df3['incomeperperson'].dropna().mean()
first = df3['incomeperperson'].quantile(q=0.2)
second = df3['incomeperperson'].quantile(q=0.4)
third  = df3['incomeperperson'].quantile(q=0.6)
fourth = df3['incomeperperson'].quantile(q=0.8)
fifth = df3['incomeperperson'].quantile(q=1)
def meangroup (row):
    if row['incomeperperson'] >=ipp_mean:
        return 1
    elif row['incomeperperson'] < ipp_mean:
        return 0
        
df3['meangroup'] = df3.apply(lambda row : meangroup(row),axis=1)
c3= df3.groupby('meangroup').size()

#cut into 5 groups
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
        
df3['income5groups'] = df3.apply(lambda row : income5groups(row),axis=1)
df3['income5groups'] = df3['income5groups'].astype('category')
# contingency table of observed counts
ct1=pd.crosstab(df3['polityscore'] , df3['income5groups'])
print (ct1)

# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print (colsum)
print(colpct)

# chi-square
print ('chi-square value, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (ct1)

#code for setting variables to numeric:
df3['polityscore'] = df3['polityscore'].convert_objects(convert_numeric=True)

# graph percent with nicotine dependence within each smoking frequency group 
seaborn.factorplot(x="income5groups", y="polityscore", data=df3, kind="bar", ci=None)
plt.xlabel('income5groups')
plt.ylabel('polityscore')
plt.show()



# Terrible vs. Bad
recode ={'1_Terrible': 'Terrible', '2_Bad':'Bad'}
df3['terriblevsbad'] = df3['income5groups'].map(recode)
ct2 = pd.crosstab(df3['polityscore'], df3['terriblevsbad'])
print ('Terrible vs. Bad\nchi-square value, p value, expected counts')
cs2= scipy.stats.chi2_contingency(ct2)
print (cs2)

# Terrible vs Average
recode1 ={'1_Terrible': 'Terrible', '3_Average':'Average'}
df3['terriblevsaverage'] = df3['income5groups'].map(recode1)
ct3 = pd.crosstab(df3['polityscore'], df3['terriblevsaverage'])
print ('Terrible vs Average\nchi-square value, p value, expected counts')
cs3= scipy.stats.chi2_contingency(ct3)
print (cs3)

# Terrible vs Decent
recode2 ={'1_Terrible': 'Terrible', '4_Decent':'Decent'}
df3['terriblevsdecent'] = df3['income5groups'].map(recode2)
ct4 = pd.crosstab(df3['polityscore'], df3['terriblevsdecent'])
print ('Terrible vs Decent\nchi-square value, p value, expected counts')
cs4= scipy.stats.chi2_contingency(ct4)
print (cs4)

#Terrible vs. Great
recode3 ={'1_Terrible': 'Terrible', '5_Great':'Great'}
df3['terriblevsgreat'] = df3['income5groups'].map(recode3)
ct5 = pd.crosstab(df3['polityscore'], df3['terriblevsgreat'])
print ('Terrible vs. Great\nchi-square value, p value, expected counts')
cs5= scipy.stats.chi2_contingency(ct5)
print (cs5)

# Bad vs Average
recode3plus ={'2_Bad': 'Bad', '3_Average':'Average'}
df3['badvsaverage'] = df3['income5groups'].map(recode3plus)
ct5plus = pd.crosstab(df3['polityscore'], df3['terriblevsgreat'])
print ('Bad vs Average\nchi-square value, p value, expected counts')
cs5plus= scipy.stats.chi2_contingency(ct5plus)
print (cs5plus)


#Bad vs. Decent
recode4 ={'2_Bad': 'Bad', '4_Decent':'Decent'}
df3['badvsdecent'] = df3['income5groups'].map(recode4)
ct6 = pd.crosstab(df3['polityscore'], df3['badvsdecent'])
print ('Bad vs. Decent\nchi-square value, p value, expected counts')
cs6= scipy.stats.chi2_contingency(ct6)
print (cs6)

# Bad vs Great
recode5 ={'2_Bad': 'Bad', '5_Great':'Great'}
df3['badvsgreat'] = df3['income5groups'].map(recode5)
ct7 = pd.crosstab(df3['polityscore'], df3['badvsgreat'])
print ('Bad vs Great\nchi-square value, p value, expected counts')
cs7= scipy.stats.chi2_contingency(ct7)
print (cs7)

#Average vs. Decent
recode5plus = {'3_Average':'Average', "4_Decent":"Decent"}
df3['averagevsdecent'] = df3['income5groups'].map(recode5plus)
ct7plus = pd.crosstab(df3['polityscore'], df3['averagevsdecent'])
print ('Average vs. Decent\nchi-square value, p value, expected counts')
cs7plus= scipy.stats.chi2_contingency(ct7plus)
print (cs7plus)


# Average vs Great
recode5plus2 = {'3_Average':'Average',  '5_Great':'Great'}
df3['averagevsgreat'] = df3['income5groups'].map(recode5plus2)
ct7plus2 = pd.crosstab(df3['polityscore'], df3['averagevsgreat'])
print ('Average vs Great\nchi-square value, p value, expected counts')
cs7plus2= scipy.stats.chi2_contingency(ct7plus2)
print (cs7plus2)



# Decent vs Great
recode6 ={'4_Decent':'Decent', '5_Great':'Great'}
df3['decentvsgreat'] = df3['income5groups'].map(recode6)
ct8 = pd.crosstab(df3['polityscore'], df3['decentvsgreat'])
print ('Decent vs Great\nchi-square value, p value, expected counts')
cs8= scipy.stats.chi2_contingency(ct8)
print (cs8)
