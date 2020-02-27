import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score,precision_score, matthews_corrcoef
from sklearn.metrics import classification_report,accuracy_score,recall_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
#from IPython.display import Image  
#pip install graphviz
#import pydotplus
#from sklearn import tree
#import collections
pwd()
unemp = pd.read_excel('E:/College/Analytics/Predictive/Unemployement.xlsx')
unemp.shape
unemp.dtypes
unemp['Civilian_labor_force_2007'].describe()
unemp.isnull().sum().sort_values(ascending=False)
unemp.groupby('State')['Unemployment_rate_2008','Unemployment_rate_2009','Unemployment_rate_2016','Unemployment_rate_2017','Unemployment_rate_2018'].mean()
unemp.groupby('State')['Median_Household_Income_2017'].mean()
#Data Preparation
unemp.fillna(unemp.mean(), inplace = True)
#t["Median_Household_Income_2017"].fillna(t["Median_Household_Income_2017"].mean(), inplace=True)
#unemp.dropna(subset=['Rural_urban_continuum_code_2013','Urban_influence_code_2013'], inplace = True)

unemp.isnull().sum()

unemp['Metro_2013'] = pd.Categorical(unemp.Metro_2013)
#t.Metro_2013 = t.Metro_2013.astype('category')

unemp.info()
print(unemp['Metro_2013'].value_counts())

#Visualization
unemp.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'),axis=1)
plt.subplots(figsize=(12,20))
p=sns.heatmap(unemp.corr(), annot=True,cmap ='RdYlGn')

f=list(['Employed_2014','Employed_2015','Unemployed_2007'])
unemp[f].corr()

k = 5 #number of variables for heatmap
corrmat=unemp.corr()
cols = corrmat.nlargest(k, 'Metro_2013')['Metro_2013'].index
unemp[cols].corr()

sns.distplot(unemp['Unemployed_2007'])
sns.distplot(unemp['Employed_2012'])
sns.distplot(unemp['Med_HH_Income_Percent_of_State_Total_2017'])
sns.distplot(unemp['Median_Household_Income_2017'])
sns.distplot(unemp['Civilian_labor_force_2018'])
sns.countplot(unemp['Metro_2013'])

sns.boxenplot(unemp['Unemployed_2018'])
sns.boxenplot(unemp['Unemployment_rate_2010'])
sns.boxenplot(unemp['Median_Household_Income_2017'])

sns.barplot(x= 'Employed_2018',y= 'Metro_2013', data = unemp)
plt.figure(figsize=(18,12))
plt.title('State Vs Unemployment 2018',fontsize=25)
sns.barplot(x= 'State',y= 'Unemployed_2018', data = unemp)

unemp.drop (columns =['FIPS','Area_name','State','Urban_influence_code_2013','Rural_urban_continuum_code_2013'], inplace = True)

x = unemp.drop(["Metro_2013"], axis=1).values 
y = unemp['Metro_2013'].cat.codes
x = preprocessing.normalize(x)

# Training, Validation and Test sets
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=615)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.2,random_state=615)

#KNN
k_range= range(3,31)
accuracy_list = []

for k in k_range:
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_val)
    accuracy_list.append(metrics.accuracy_score(y_val,y_pred))

print(accuracy_list)

knn= KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

print(y_test.value_counts())
pd.crosstab(y_pred,y_test, rownames=['Predicted'], colnames=['True'], margins=True)
matthews_corrcoef(y_pred,y_test)
confusion_matrix(y_pred,y_test)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)

#Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred_nb = nb.predict(x_test)
pd.crosstab(y_pred_nb, y_test, rownames=['Predicted'], colnames=['True'], margins=True)
matthews_corrcoef(y_test,y_pred_nb)
accuracy_score(y_test,y_pred_nb)
precision_score(y_test,y_pred_nb)
recall_score(y_test,y_pred_nb)
confusion_matrix(y_test,y_pred_nb)

#Bagged Naive Bayes
nb_bag=BaggingClassifier(GaussianNB(),random_state=96,max_samples =20,max_features=9)
nb_bag.fit(x_train, y_train)
nb_pred_bag = nb_bag.predict(x_test)
pd.crosstab(nb_pred_bag, y_test, rownames=['Predicted'], colnames=['True'], margins=True)
matthews_corrcoef(y_test,nb_pred_bag)
print(accuracy_score(y_test,nb_pred_bag))
precision_score(y_test,nb_pred_bag)
recall_score(y_test,nb_pred_bag)

#Decision Tree
grid= {"min_samples_leaf" : [1,2,3,4,5],"criterion":["gini","entropy"],"max_depth":[3,5,7,9], "max_features":[6,7,8]}
dt=DecisionTreeClassifier(random_state=96)
gridsearch= GridSearchCV(dt,param_grid=grid,cv=10)
gridsearch.fit(x_val, y_val)
print(gridsearch.best_score_)
print(gridsearch.best_params_)

dt=DecisionTreeClassifier(criterion= 'entropy', max_depth= 3, min_samples_leaf= 1,max_features=6,random_state=96)
dt.fit(x_train, y_train)
dt_pred= dt.predict(x_test)
pd.crosstab(dt_pred,y_test, rownames=['Predicted'], colnames=['True'], margins=True)
matthews_corrcoef(y_test,dt_pred)
precision_score(y_test,dt_pred)
recall_score(y_test,dt_pred)
accuracy_score(y_test,dt_pred)
roc_auc_score(y_test,dt_pred)
confusion_matrix(y_test,dt_pred)

#dot_data = tree.export_graphviz(dt, out_file=None, feature_names=features, 
#                filled=True, rounded=True,
#                special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data)  
#colors = ('turquoise', 'orange')
#edges = collections.defaultdict(list)
#for edge in graph.get_edge_list():
#    edges[edge.get_source()].append(int(edge.get_destination()))
#for edge in edges:
#    edges[edge].sort()    
#    for i in range(2):
#        dest = graph.get_node(str(edges[edge][i]))[0]
#        dest.set_fillcolor(colors[i])
#graph.write_png('tree.png')
#Image(graph.create_png())

#Random Forest
par_grid= {"min_samples_leaf" : [1,2,3,4,5],"criterion":["gini","entropy"],"n_estimators":[20,40,60,80,100],"max_depth":[3,5,7,9],"max_features":[6,7,8]}
rf=RandomForestClassifier(random_state=96)
grid_search= GridSearchCV(rf,param_grid=par_grid,cv=10)
grid_search.fit(x_val, y_val)
print(grid_search.best_score_)
print(grid_search.best_params_)

rf=RandomForestClassifier(criterion= 'gini', max_depth= 9,n_estimators=60 ,min_samples_leaf= 2,max_features=7,random_state=96)
rf.fit(x_train, y_train)
rf_pred= rf.predict(x_test)
pd.crosstab(rf_pred, y_test,  rownames=['Predicted'], colnames=['True'], margins=True)
matthews_corrcoef(y_test,rf_pred)
precision_score(y_test,rf_pred)
recall_score(y_test,rf_pred)
accuracy_score(y_test,rf_pred)
roc_auc_score(y_test,rf_pred)
print(classification_report(y_test,rf_pred))
confusion_matrix(y_test,rf_pred)

features = list(unemp.columns[:53])
del features[2]

tmp=pd.DataFrame({'Feature': features,'Feature importance': rf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False).head(10)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
s.set_title('Top 10 Features importance',fontsize=20)


