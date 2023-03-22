#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df=pd.read_excel(r'C:\Users\akank\OneDrive\Desktop\Dataset_python\Finance\Bank_Personal_Loan_Modelling.xlsx',1)
df.head()


# In[10]:


df.shape


# In[11]:


df.isnull().sum()


# In[12]:


df.drop(['ID','ZIP Code'],axis=1,inplace=True)


# In[14]:


df.columns


# In[15]:


import plotly.express as px


# In[16]:


fig=px.box(df,y=['Age', 'Experience', 'Income', 'Family', 'Education'])
fig.show()


# In[17]:


df.skew()


# In[ ]:


df.dtypes


# In[18]:


df.hist(figsize=(20,20))


# In[ ]:


import seaborn as sns


# In[20]:


import warnings
warnings.filterwarnings('ignore')


# In[21]:


sns.distplot(df['Experience'])


# In[22]:


df['Experience'].mean()


# In[23]:


Negative_exp=df[df['Experience']<0]
Negative_exp.head()


# In[24]:


sns.distplot(Negative_exp['Age'])


# In[ ]:


Negative_exp['Experience'].mean()


# In[ ]:


Negative_exp.size


# In[25]:


print('There are {} records which has negative values for experience, approx {} %'.format(Negative_exp.size , ((Negative_exp.size/df.size)*100)))


# In[26]:


data=df.copy()
data.head()


# In[27]:


data['Experience']=np.where(data['Experience']<0,data['Experience'].mean(),data['Experience'])
data[data['Experience']<0]


# In[28]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)


# In[ ]:


#We could see that Age & Experience are very strongly correlated,
#Hence it is fine for us to go with Age and drop Experience to avoid multi-colinearity issue.


# In[29]:


data=data.drop(['Experience'],axis=1)
data.head()


# In[30]:


data['Education'].unique()


# In[31]:


def mark(x):
    if x==1:
        return 'Undergrad'
    elif x==2:
        return 'Graduate'
    else:
        return 'Advanced/Professional'


# In[32]:


data['Edu_mark']=data['Education'].apply(mark)
data.head()


# In[33]:


EDU_dis=data.groupby('Edu_mark')['Age'].count()
EDU_dis


# In[35]:


fig=px.pie(data,values=EDU_dis, names=EDU_dis.index,title='Pie Chart')
fig.show()


# In[ ]:


#Inference :We could see that We have more Undergraduates 41.92% than graduates(28.06%) & Advanced Professional(30.02%)


# In[36]:


data.columns


# In[ ]:


#Lets Explore the account holder's distribution


# In[37]:


def Security_CD(row):
    if (row['Securities Account']==1) & (row['CD Account']==1):
        return 'Holds Securites & Deposit'
    elif (row['Securities Account']==0) & (row['CD Account']==0):
        return 'Does not Holds Securites or Deposit'
    elif (row['Securities Account']==1) & (row['CD Account']==0):
        return ' Holds only Securites '
    elif (row['Securities Account']==0) & (row['CD Account']==1):
        return ' Holds only Deposit'


# In[38]:


data['Account_holder_category']=data.apply(Security_CD,axis=1)
data.head()


# In[39]:


values=data['Account_holder_category'].value_counts()
values.index


# In[41]:


fig=px.pie(data,values=values, names=values.index,title='Pie Chart')
fig.show()


# In[42]:


#We could see that alomst 87% of customers do not hold any securities or deposit, and 3 % hold both securities as well as deposit. It will be good if we encourage those 87% to open any of these account as it will improve the assests of the bank


# In[43]:


data.columns


# In[44]:


px.box(data,x='Education',y='Income',facet_col='Personal Loan')


# In[ ]:


#Inference:From the above plot we could say that Income of customers who availed personal loan are alomst same irrescpective of their Education


# In[45]:


plt.figure(figsize=(12,8))
sns.distplot(data[data['Personal Loan']==0]['Income'],hist=False,label='Income with no personal loan')
sns.distplot(data[data['Personal Loan']==1]['Income'],hist=False,label='Income with personal loan')
plt.legend()


# In[ ]:


#Conclusion: Customers Who have availed personal loan seem to have higher income than those who do not have personal loan automate above stuffs


# In[46]:


def plot(col1,col2,label1,label2,title):
    plt.figure(figsize=(12,8))
    sns.distplot(data[data[col2]==0][col1],hist=False,label=label1)
    sns.distplot(data[data[col2]==1][col1],hist=False,label=label2)
    plt.legend()
    plt.title(title)


# In[47]:


plot('Income','Personal Loan','Income with no personal loan','Income with personal loan','Income Distribution')


# In[48]:


plot('CCAvg','Personal Loan','Credit card avg with no personal loan','Credit card avg with personal loan','Credit card avg Distribution')


# In[49]:


plot('Mortgage','Personal Loan','Mortgage of customers with no personal loan','Mortgage of customers  with personal loan','Mortgage of customers  Distribution')


# In[ ]:


#People with high mortgage value, i.e more than 400K have availed personal Loan


# In[50]:


data.columns


# In[51]:


col_names=['Securities Account','Online','Account_holder_category','CreditCard']


# In[52]:


for i in col_names:
    plt.figure(figsize=(10,5))
    sns.countplot(x=i,hue='Personal Loan',data=data)


# In[ ]:


#From the above graph we could infer that , customers who hold deposit account & customers who do not hold either a securities account or deposit account have aviled personal loan

#Perform Hypothesis Testing

#Q.. How Age of a person is going to be a factor in availing loan ??? Does Income of a person have an impact on availing loan ??? Does the family size makes them to avail loan ???¶


# In[53]:


sns.scatterplot(data['Age'],data['Personal Loan'],hue=data['Family'])


# In[56]:


import scipy.stats as stats


# In[57]:


Ho='Age does not have impact on availing personal loan'
Ha='Age does  have impact on availing personal loan'


# In[58]:


Age_no=np.array(data[data['Personal Loan']==0]['Age'])
Age_yes=np.array(data[data['Personal Loan']==1]['Age'])


# In[59]:


t,p_value=stats.ttest_ind(Age_no,Age_yes,axis=0)
if p_value<0.05:
    print(Ha,' as the p_value is less than 0.05 with a value of {}'.format(p_value))
else:
    print(Ho,' as the p_value is greater than 0.05 with a value of {}'.format(p_value))


# In[ ]:


#automate above stuffs


# In[60]:


def Hypothesis(col1,col2,HO,Ha):
    arr1=np.array(data[data[col1]==0][col2])
    arr2=np.array(data[data[col1]==1][col2])
    t,p_value=stats.ttest_ind(arr1,arr2,axis=0)
    if p_value<0.05:
        print('{}, as the p_value is less than 0.05 with a value of {}'.format(Ha,p_value))
    else:
        print('{} as the p_value is greater than 0.05 with a value of {}'.format(HO,p_value))


# In[61]:


Hypothesis('Personal Loan','Age',HO='Age does not have impact on availing personal loan',Ha='Age does  have impact on availing personal loan')


# In[ ]:


#Q..Income of a person has significant impact on availing Personal Loan or not?


# In[62]:


Hypothesis(col1='Personal Loan',col2='Income',HO='Income does not have impact on availing personal loan',Ha='Income does  have impact on availing personal loan')


# In[ ]:


#Income have phenomenal significance on availing personal Loan , As the P_value is less than 0.05 with a value of :0.0
#Q..Number of persons in the family has significant impact on availing Personal Loan or not?


# In[63]:


Hypothesis('Personal Loan','Family',HO='AgFamily does not have impact on availing personal loan',Ha='Family does  have impact on availing personal loan')


# In[ ]:


#Family have phenomenal significance on availing personal Loan , As the P_value is less than 0.05 with a value of :1.4099040685673807e-05

