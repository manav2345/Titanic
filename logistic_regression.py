#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("D:\IBM\Titanic.csv")
print("Total passangers",len(data.index))
#%%
sns.countplot(x="Survived",hue="Sex",data=data)
#%%
data["Age"].plot.hist()
#%%
sns.boxplot(x="Pclass",y="Age",data=data)
#%%
plt.scatter(data['Survived'],data['Age'])
plt.show()
#%%
sns.distplot(data["Age"].dropna())
#%%
data.info()
#%%
data.isnull().sum()
#%%
data.drop("Cabin",axis=1,inplace=True)
data["Age"].fillna(data["Age"].median(),inplace=True)
data.dropna(inplace=True)
s=pd.get_dummies(data["Sex"],drop_first=True)
e=pd.get_dummies(data["Embarked"])
data=pd.concat([data,s,e],axis=1)
#%%
data.drop(["Sex","Embarked","PassengerId","Pclass","Name","Ticket"],axis=1,inplace=True)
data.head(10)
#%%
from sklearn import linear_model,datasets
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
x=data.drop("Survived",axis=1)
y=data["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
#Logistic Regression
reg = linear_model.LogisticRegression()
reg.fit(x_train,y_train)
pri=reg.predict(x_test)
print("Logistic Regression\n",accuracy_score(y_test,pri)*100)
#print(reg.score(x_train,y_train)*100)
#Random forest
from sklearn.ensemble import  RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100000)
random_forest.fit(x_train, y_train)
pri = random_forest.predict(x_test)
print("Random forest\n",accuracy_score(y_test,pri)*100)
#print(random_forest.score(x_train,y_train)*100)
#%%
test=pd.read_csv("Titanic Test.csv")
print("Passangers to pridict",len(test.index))
p=pd.read_csv("Titanic Predictions .csv")
print("Pridiction of passangers",len(p.index))
#%%
test["Age"].fillna(test["Age"].median(),inplace=True)
s=pd.get_dummies(test["Sex"],drop_first=True)
e=pd.get_dummies(test["Embarked"])
test=pd.concat([test,s,e],axis=1)
test.drop(["Sex","Embarked","Pclass","Name","Ticket","Cabin"],axis=1,inplace=True)
#%%
pri=reg.predict(test.drop("PassengerId",axis=1).fillna(True))
print("Logistic Regression\n",accuracy_score(p.drop("PassengerId",axis=1),pri)*100)
pri = random_forest.predict(test.drop("PassengerId",axis=1).fillna(False))
print("Random forest\n",accuracy_score(p.drop("PassengerId",axis=1),pri)*100)
#%% Student T test
null=data[data.Age.isnull()]
print("Null age ",len(null.index))
#%%
from scipy import stats
sp1=data.Age.dropna().sample(30)#take 30 sample allways
sp2=data.Age.dropna().sample(30)
print("mean of sp1",sp1.mean(),"mean of sp2",sp2.mean())
stat,pvalue=stats.ttest_ind(sp1,sp2)
print("Stat",stat,"pvalue%",pvalue*100)
if(pvalue>0.05):
    print("Valid")
else:
    print("Invalid")
#%%
table=pd.crosstab(data.Sex,data.Survived)
print(table)
#%%
g,p,dof,expctd=stats.chi2_contingency(table)
if(p):
#%%


# %%
