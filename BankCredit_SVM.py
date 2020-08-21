#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import preprocessing
import numpy as np
import pandas as pd


# In[3]:


df1=pd.read_csv("C:\\Users\\Sanika\\Desktop\\Dataset\\Dataset\\BankCreditCard.csv")


# In[4]:


df1.head()


# In[5]:


df1.info()


# In[6]:


y=df1['Default_Payment']
x=df1.drop('Default_Payment',axis=1)


# In[8]:


df1.columns


# In[13]:


from sklearn import preprocessing
MyScalar=preprocessing.StandardScaler()
Scalar=MyScalar.fit(x)
df_x=pd.DataFrame(Scalar.transform(x),columns=(['Customer ID', 'Credit_Amount', 'Gender', 'Academic_Qualification',
       'Marital', 'Age_Years', 'Repayment_Status_Jan', 'Repayment_Status_Feb',
       'Repayment_Status_March', 'Repayment_Status_April',
       'Repayment_Status_May', 'Repayment_Status_June', 'Jan_Bill_Amount',
       'Feb_Bill_Amount', 'March_Bill_Amount', 'April_Bill_Amount',
       'May_Bill_Amount', 'June_Bill_Amount', 'Previous_Payment_Jan',
       'Previous_Payment_Feb', 'Previous_Payment_March',
       'Previous_Payment_April', 'Previous_Payment_May',
       'Previous_Payment_June']))


# In[14]:


x=df_x


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)


# In[16]:


######linear svm
from sklearn import svm


# In[17]:


clf=svm.SVC(C=1.0,kernel='linear')
clf.fit(x_train,y_train)


# In[18]:


y_pred=clf.predict(x_test)


# In[19]:


from sklearn import metrics as mt


# In[20]:


print("Accuracy =%.2f"%mt.accuracy_score(y_test,y_pred))


# In[22]:


clf=svm.SVC(C=1.0,kernel='rbf')
clf.fit(x_train, y_train)


# In[23]:


y_pred=clf.predict(x_test)


# In[24]:


print("Accuracy =%.2f"%mt.accuracy_score(y_test,y_pred))


# In[25]:


clf=svm.SVC(C=1.0,kernel='poly',degree=2)
clf.fit(x_train, y_train)


# In[26]:


y_pred=clf.predict(x_test)


# In[27]:


print("Accuracy =%.2f"%mt.accuracy_score(y_test,y_pred))


# In[28]:


df2=df1.drop('Customer ID',axis=1)


# In[29]:


df = df2[:df2.shape[0]]
df['Default_Payment'] = df2['Default_Payment'].values
corr_values_df = pd.DataFrame(df.corr()['Default_Payment'].abs().sort_values(ascending=True))
print(corr_values_df[0:10])


# In[30]:


df2.columns


# In[31]:


miss_features=corr_values_df[0:10].index
print(miss_features)


# In[32]:


df2.drop(miss_features,inplace=True,axis=1)


# In[33]:


df2.head()


# In[34]:


y=df2['Default_Payment']
x=df2.drop('Default_Payment',axis=1)


# In[35]:


df2.columns


# In[36]:


from sklearn import preprocessing
MyScalar=preprocessing.StandardScaler()
Scalar=MyScalar.fit(x)
df_x=pd.DataFrame(Scalar.transform(x),columns=(['Credit_Amount', 'Repayment_Status_Jan', 'Repayment_Status_Feb',
       'Repayment_Status_March', 'Repayment_Status_April',
       'Repayment_Status_May', 'Repayment_Status_June', 'Previous_Payment_Jan',
       'Previous_Payment_Feb', 'Previous_Payment_March',
       'Previous_Payment_April', 'Previous_Payment_May',
       'Previous_Payment_June']))


# In[37]:


x=df_x


# In[38]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)


# In[40]:


clf=svm.SVC(kernel='linear',C=1.0)
clf.fit(x_train,y_train)


# In[41]:


y_pred=clf.predict(x_test)


# In[44]:


print("Accuracy =%2f"%mt.accuracy_score(y_test,y_pred))


# In[45]:


clf=svm.SVC(C=1.0,kernel='rbf')
clf.fit(x_train, y_train)


# In[46]:


y_pred=clf.predict(x_test)


# In[47]:


print("Accuracy =%.2f"%mt.accuracy_score(y_test,y_pred))


# In[48]:


clf=svm.SVC(C=1.0,kernel='poly',degree=2)
clf.fit(x_train, y_train)


# In[49]:


y_pred=clf.predict(x_test)


# In[50]:


print("Accuracy =%2f"%mt.accuracy_score(y_test,y_pred))


# In[ ]:




