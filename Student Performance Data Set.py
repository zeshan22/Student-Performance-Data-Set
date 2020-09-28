#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# In[2]:


stu_por=pd.read_csv('student-mat.csv',sep=';')
stu_mat=pd.read_csv('student-por.csv',sep=';')


# In[3]:


stu =pd.concat([stu_por,stu_mat])


# In[4]:


stu.head()


# In[5]:


stu['total_grades'] = (stu['G1']+stu['G2']+stu['G3'])/3


# In[6]:


stu=stu.drop(["G1","G2","G3"],axis=1)


# In[7]:


stu.head()


# In[8]:


stu["total_grades"].max(),stu["total_grades"].min()


# In[9]:


#ranging the grade in three parts
def marks(total_grades):
    if(total_grades<7):
        return("low")
    elif(total_grades>=7 and total_grades<14):
        return("average")
    elif(total_grades>=14):
        return("high")
stu["grades"]=stu["total_grades"].apply(marks)


# In[10]:


stu.head()


# In[11]:


stu.describe()


# In[12]:


stu.dtypes


# In[13]:


stu.info()


# In[14]:


#checking for null values
stu.isnull().any()


# In[15]:


#visualizing the grades
plt.figure(figsize=(8,6))
sns.countplot(stu["grades"], order=["low","average","high"], palette='Set1')
plt.title('Final Grade - Number of Students',fontsize=20)
plt.xlabel('Final Grade', fontsize=16)
plt.ylabel('Number of Student', fontsize=16)


# In[16]:


#describing correlation
corr=stu.corr()

plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True, cmap="Reds")
plt.title('Correlation Heatmap', fontsize=20)


# In[17]:


#ANALYZING CATEGORICAL VARIABLES


# In[18]:



#comparing school with grades
sns.boxplot(x="school", y="total_grades", data=stu)

school_counts=stu["school"].value_counts().to_frame()
school_counts.rename(columns={"school":"school_counts"},inplace=True)
school_counts.index.name='school'

school_sns=sns.countplot(hue=stu["school"],x=stu["grades"],data=stu)

#crosstab is expanded form of value counts the the factors inside any variables
perc=(lambda col:col/col.sum())
index=["average","high","low"]
schooltab1=pd.crosstab(columns=stu.school,index=stu.grades)

school_perc=schooltab1.apply(perc).reindex(index)

school_perc.plot.bar(colormap="PiYG_r",fontsize=15,figsize=(7,7))
plt.title('Final Grade By school', fontsize=20)
plt.ylabel('Percentage of Student Counts ', fontsize=16)
plt.xlabel('Final Grade', fontsize=16)
plt.show()

#so by graph we know that school has impact on grades of students


# In[19]:


sns.boxplot(x="school", y="total_grades", data=stu)


# In[20]:


#comparing sex with grades
sns.boxplot(x="sex", y="total_grades", data=stu)
school_counts=stu["sex"].value_counts()


stu[["sex","total_grades"]].groupby(["sex"],as_index=False).mean()
#as the graph of sex nearly overlaps so it will not have impact on grades
stu=stu.drop(["sex"],axis=1)


# In[21]:


#comparing address with grades
sns.boxplot(x="address", y="total_grades", data=stu)
index=["average","high","low"]
addresstab1=pd.crosstab(columns=stu.address,index=stu.grades)

address_perc=addresstab1.apply(perc).reindex(index)

address_perc.plot.bar(colormap="PiYG_r",fontsize=15,figsize=(7,7))
plt.title('Final Grade By address', fontsize=20)
plt.ylabel('Percentage of Student Counts ', fontsize=16)
plt.xlabel('Final Grade', fontsize=16)
plt.show()
#address is factor for the grades


# In[22]:


#comparing famsize with grades
sns.boxplot(x="famsize", y="total_grades", data=stu)
famsizetab1=pd.crosstab(columns=stu.famsize,index=stu.grades)

famsize_perc=famsizetab1.apply(perc).reindex(index)

famsize_perc.plot.bar(colormap="PiYG_r",fontsize=15,figsize=(7,7))
plt.title('Final Grade By famsize', fontsize=20)
plt.ylabel('Percentage of Student Counts ', fontsize=16)
plt.xlabel('Final Grade', fontsize=16)
plt.show()
#famsize has great impact on grades


# In[23]:


#comparing pstatus with grades
sns.boxplot(x="Pstatus", y="total_grades", data=stu)
Pstatustab1=pd.crosstab(columns=stu.Pstatus,index=stu.grades)

Pstatus_perc=Pstatustab1.apply(perc).reindex(index)

Pstatus_perc.plot.bar(colormap="PiYG_r",fontsize=15,figsize=(7,7))
plt.title('Final Grade By Pstatus', fontsize=20)
plt.ylabel('Percentage of Student Counts ', fontsize=16)
plt.xlabel('Final Grade', fontsize=16)
plt.show()



stu[["Pstatus","total_grades"]].groupby(["Pstatus"],as_index=False).mean()
#it is not a good factor


# In[24]:


#comparing jobs
sns.boxplot(x="Mjob", y="total_grades", data=stu)
sns.boxplot(x="Fjob", y="total_grades", data=stu)
stu1=stu[["Fjob","Mjob","total_grades"]]
job_grp=stu1.groupby(['Mjob','Fjob'],as_index=False).mean()
job_pivot=job_grp.pivot(index='Mjob',columns='Fjob',values='total_grades')

#so father and mother jobs has great impact on grades


# In[25]:


#comparing reasons
sns.boxplot(x="reason", y="total_grades", data=stu)
#it has impact on the grades


# In[26]:


#comparing guardians
sns.boxplot(x="guardian", y="total_grades", data=stu)

guardiantab1=pd.crosstab(columns=stu.guardian,index=stu.grades)
guardian_perc=guardiantab1.apply(perc).reindex(index)
guardian_perc.plot.bar(colormap="BrBG",fontsize=15,figsize=(7,7))
plt.title('Final Grade By guardian', fontsize=20)
plt.ylabel('Percentage of Student Counts ', fontsize=16)
plt.xlabel('Final Grade', fontsize=16)
plt.show()
#so guardian has grat impact on grades


# In[27]:


#support of family and school
sns.boxplot(x="schoolsup", y="total_grades", data=stu)
#it is the important factor
sns.boxplot(x="famsup", y="total_grades", data=stu)
stu[["famsup","total_grades"]].groupby(["famsup"],as_index=False).mean()

#stu[["schoolsup","total_grades"]].groupby(["schoolsup"],as_index=False).mean()
#famsup does not have great impact on grades 
stu=stu.drop(["famsup"],axis=1) 


# In[31]:


#comparing paid attributes
sns.boxplot(x="paid", y="total_grades", data=stu)
paidtab1=pd.crosstab(columns=stu.paid,index=stu.grades)
paid_perc=paidtab1.apply(perc).reindex(index)
paid_perc.plot.bar(colormap="BrBG",fontsize=15,figsize=(7,7))
plt.title('Final Grade By paid', fontsize=20)
plt.ylabel('Percentage of Student Counts ', fontsize=16)
plt.xlabel('Final Grade', fontsize=16)
plt.show()
#paid does not have much influence on grades so
stu=stu.drop(["paid"],axis=1)

#stu[["paid","total_grades"]].groupby(["paid"],as_index=False).mean()


# In[32]:


sns.boxplot(x="activities", y="total_grades", data=stu)
#is has graet impact on student perforamnce
sns.boxplot(x="nursery", y="total_grades", data=stu)
#it does not have great impact on performance
stu=stu.drop(["nursery"],axis=1)


# In[33]:


#comparing if higher educatiob of students have impact on performance
sns.boxplot(x="higher", y="total_grades", data=stu)

sns.boxplot(x="internet", y="total_grades", data=stu)
#internet also have great impact on performance of individual


# In[34]:


#high school romace impact on the performance of students
sns.boxplot(x="romantic", y="total_grades", data=stu)
romantictab1=pd.crosstab(columns=stu.romantic,index=stu.grades)
romantic_perc=romantictab1.apply(perc).reindex(index)
romantic_perc.plot.bar(colormap="BrBG",fontsize=15,figsize=(7,7))
plt.title('Final Grade By romantic', fontsize=20)
plt.ylabel('Percentage of Student Counts ', fontsize=16)
plt.xlabel('Final Grade', fontsize=16)
plt.show()
#so high school romance leads to decline in performance of students
#beware of that


# In[35]:


stu[["romantic","total_grades"]].groupby(["romantic"],as_index=False).mean()


# In[36]:


stu.columns


# In[38]:


stu.info()


# In[39]:


stu1=pd.get_dummies(stu,columns=["school","address","famsize","Pstatus","Mjob","Fjob","reason","guardian", 'schoolsup', 'activities', 'higher', 'internet', 'romantic' ])


# In[40]:


stu1.head()


# In[41]:


#test data
test_stu1=stu1["grades"]
teststu1=stu1["total_grades"]
#train data
train_stu1=stu1.drop(['total_grades','grades'],axis=1)
train_stu=train_stu1.values


# In[43]:


#ANALYZING NUMERICAL VARIABLES
from scipy import stats
#comparing age with marks
sns.regplot(x="age",y="total_grades",data=stu)


# In[44]:


#pearson coeffiecient
stu[["age","total_grades"]].corr()


# In[45]:


#p-value
pearson_coef , p_value=stats.pearsonr(stu["age"],stu["total_grades"])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#age is not a good factor


# In[46]:



#if p-value is greater than 0.6 than we will removethat feature
import statsmodels.api as sm
X=np.append(arr=np.ones((1044,1)).astype(int),values=train_stu,axis=1)
X_opt = X[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13]]
regressor_ols=sm.OLS(endog=teststu1,exog=X_opt).fit()
regressor_ols.summary()

X_opt = X[:, [0,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor_ols=sm.OLS(endog=teststu1,exog=X_opt).fit()
regressor_ols.summary()

X_opt = X[:, [0,2,3,4,5,6,7,9,10,11,12,13]]
regressor_ols=sm.OLS(endog=teststu1,exog=X_opt).fit()
regressor_ols.summary()


# In[47]:


#now we merge our training data
train_x=np.concatenate((X_opt,X[:,14:49]),axis=1)


# In[48]:


stu[["Medu","total_grades"]].corr()


# In[49]:



stu[["Fedu","total_grades"]].corr()


# In[50]:


train_stu2=train_stu1.drop(["age","freetime"],axis=1)
np1=[1 for i in range(0,1044)]
train_stu2.insert(loc=0,column= "noimprotance", value=np1)
#now after getting the proper features we will split the data


# In[51]:


train_stu2.head()


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x,test_stu1, test_size = 0.2, random_state = 0)


# In[53]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=80,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

#predicting the test set re4sults
y_pred_random=classifier.predict(X_test)


# In[54]:


importances=classifier.feature_importances_


# In[55]:


importances


# In[56]:


importances=classifier.feature_importances_
for i,features in zip(importances,[ 'noimportant','Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'goout', 'Dalc', 'Walc', 'health', 'absences',
       'total_grades', 'grades', 'school_GP', 'school_MS', 'address_R',
       'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T',
       'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services',
       'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other',
       'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home',
       'reason_other', 'reason_reputation', 'guardian_father',
       'guardian_mother', 'guardian_other', 'schoolsup_no', 'schoolsup_yes',
       'activities_no', 'activities_yes', 'higher_no', 'higher_yes',
       'internet_no', 'internet_yes', 'romantic_no', 'romantic_yes']):
    print("{}:{}".format(features,i))


# In[57]:


indices = np.argsort(importances)
indices


# In[58]:


# Rearrange feature names so they match the sorted feature importances
names = [train_stu2.columns[i] for i in indices]
names


# In[59]:


# Barplot: Add bars

plt.figure(figsize=(20,20))
plt.bar(range(train_x.shape[1]), importances[indices],width=0.5)
# Add feature names as x-axis labels
plt.xticks(range(train_x.shape[1]),names, rotation=60, fontsize = 12)
#from here we cam see that absences is the important features for determining the grades of students

# Create plot title
plt.title("Feature Importance")
# Show plot
plt.show()


# In[60]:


#determinnig the confusion matrix
from sklearn.metrics import confusion_matrix
cm_random=confusion_matrix(y_test,y_pred_random)

#determining the precision,recall and f1-score 
from sklearn.metrics import classification_report
report_random=classification_report(y_test,y_pred_random)


# In[62]:





# In[64]:





# In[65]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_SVC= classifier.predict(X_test)
cm_SVC=confusion_matrix(y_test,y_pred_SVC)

#determining the precision,recall and f1-score 
from sklearn.metrics import classification_report
report_SVC=classification_report(y_test,y_pred_SVC)
print(report_SVC)


# In[66]:


#fitting logistic regression to the training set

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred_logistic= classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_logistic= confusion_matrix(y_test, y_pred_logistic)

#determining the precision,recall and f1-score 
from sklearn.metrics import classification_report
report_logistic=classification_report(y_test,y_pred_logistic)


# In[67]:


#fitting the knn_calssifier to the training set


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred_knn= classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn= confusion_matrix(y_test, y_pred_knn)

#determining the precision,recall and f1-score 
from sklearn.metrics import classification_report
report_knn=classification_report(y_test,y_pred_knn)


# In[68]:


print(report_random)
print(report_SVC)
print(report_logistic)
print(report_knn)


# In[69]:


#SO BY CONFUSION MATRIX AND F-SCORE WE FIND OUT THAT RANDOM FOREST IS BEST CLASSIFIER FOR GIVEN PROBLEM.


# In[ ]:




