import pandas as pd
import numpy as np

import seaborn as sns

import statsmodels.api as sm
import pylab as py
import scipy
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
## now reading the dataset into python by using pandas


df= pd.read_csv(r"C:\Users\ssindhu\Downloads\archive (2)\Admission_Predict_Ver1.1.csv")


df
## displays the top 5 rows of the dataset
df.head()
df.info
## gives the description of each  numeric variable in the dataset
df1=df.describe()

## data preprocessing

## checking for the datatypes of the variables
df.dtypes## as the  all variables are in appropriate data types there is no need of typecasting


## checking for duplicates in the dataset

duplicate = df.duplicated()
duplicate

sum(duplicate) ## there are no duplicates present in the dataset


# Duplicates in Columns
# We can use correlation coefficient values to identify columns which have duplicate information

d1 = df.corr()



## missing values

# Check for count of NA's in each column
df.isna().sum()

df=df.rename(columns={'Serial No.':'no','GRE Score':'gre','TOEFL Score':'toefl','University Rating':'rating','SOP':'sop','LOR ':'lor',
                           'CGPA':'gpa','Research':'research','Chance of Admit ':'chance'})

df.columns

df.dtypes

df.groupby('rating').mean()

df[df['chance']>0.82].groupby('chance').mean()

## outliers 
## plotiing boxplot to check otliers are present or not
import matplotlib.pyplot as plt
sns.boxplot(df['no']) 

sns.boxplot(df['gre'])

sns.boxplot(df['toefl']) 
 
sns.boxplot(df['rating']) 

sns.boxplot(df['sop'])

sns.boxplot(df['lor'])

sns.boxplot(df['gpa'])

sns.boxplot(df['research)'])
sns.boxplot(df['chance']) 



## EDA

## auto eda
import sweetviz as sv


s = sv.analyze(df)
s.show_html()


##business moments

## first business moments
## measure of central trendency

df.mean()
df.median()


## measure of dispersion
df.var()
df.std()


## skewness
df.skew()


## kurtosis

df.kurt()


## graphical presentation

## univariate for numeric features

## histogram

sns.histplot(df['no']) 
sns.histplot(df['gre'])
sns.histplot(df['toefl']) 
sns.histplot(df['rating']) 
sns.histplot(df['sop'])
sns.histplot(df['lor']) 
sns.histplot(df['gpa']) 
sns.histplot(df['research'])  
sns.histplot(df['chance'])

 
## Q-Q plot


sm.qqplot(df['no'],fit=True,line='45')

sm.qqplot(df['gre'],fit=True,line='45') 

sm.qqplot(df['toefl'],fit=True,line='45')  

sm.qqplot(df['rating'],fit=True,line='45') 

sm.qqplot(df['sop'],fit=True,line='45')  

sm.qqplot(df['lor'],fit=True,line='45') 

sm.qqplot(df['gpa'],fit=True,line='45')  

sm.qqplot(df['research'],fit=True,line='45')  

sm.qqplot(df['chance'],fit=True,line='45')  

sns.pairplot(df)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Drop the 'no' column from the DataFrame
df.drop(['no'],axis=1, inplace=True)

# Separate the target variable and features
cy = df['chance']
X = df.drop('chance', axis=1)


# Split into training and testing sets
X_train, X_test, cy_train, cy_test = train_test_split(X, cy, test_size=0.2, random_state=0)

# Perform feature scaling using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Convert the y_train and y_test to binary labels
cy_train = np.where(cy_train > 0.83, 1, 0).astype(int)
cy_test = np.where(cy_test > 0.83, 1, 0).astype(int)

# Logistic Regression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, cy_train)
y_pred_lr = logreg.predict(X_test)

from sklearn.linear_model import LogisticRegression

# Create a logistic regression model object
logreg = LogisticRegression()

# Train the model on the training data
logreg.fit(X_train, cy_train)
log_train = round(logreg.score(X_train, cy_train) * 100, 2)
log_accuracy = round(accuracy_score(y_pred_lr, cy_test) * 100, 2)


print("Training Accuracy    :", log_train, "%")
print("Model Accuracy Score :", log_accuracy, "%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n", classification_report(cy_test, y_pred_lr))
print("\033[1m--------------------------------------------------------\033[0m")
plot_confusion_matrix(logreg, X_test, cy_test)
plt.title('Confusion Matrix')


###SVM
# Support Vector Machines
from sklearn.svm import SVC
svc = SVC()

# Convert y_test to binary labels
cy_test_binary = np.where(cy_test > 0.83, 1, 0)

svc.fit(X_train, cy_train)
y_pred_svc = svc.predict(X_test)

svc_train = round(svc.score(X_train, cy_train) * 100, 2)
svc_accuracy = round(accuracy_score(y_pred_svc, cy_test) * 100, 2)

print("Training Accuracy    :", svc_train, "%")
print("Model Accuracy Score :", svc_accuracy, "%")
print("\033[1m--------------------------------------------------------\033[0m")
classification_report(cy_test_binary, y_pred_svc)
print("\033[1m--------------------------------------------------------\033[0m")
plot_confusion_matrix(svc, X_test, cy_test)
plt.title('Confusion Matrix')


###Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
cy_test_binary = np.where(cy_test > 0.83, 1, 0)

random_forest.fit(X_train, cy_train)
y_pred_rf = random_forest.predict(X_test)
random_forest.score(X_train, cy_train)

random_forest_train = round(random_forest.score(X_train, cy_train) * 100, 2)
random_forest_accuracy = round(accuracy_score(y_pred_rf, cy_test) * 100, 2)

print("Training Accuracy    :", random_forest_train, "%")
print("Model Accuracy Score :", random_forest_accuracy, "%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n", classification_report(cy_test, y_pred_rf))
print("\033[1m--------------------------------------------------------\033[0m")
plot_confusion_matrix(random_forest, X_test, cy_test)
plt.title('Confusion Matrix')




models = pd.DataFrame({
    'Model': [
        'Support Vector Machines', 'Logistic Regression', 'Random Forest',
            ],

    'Training Accuracy':
    [log_train, svc_train,  random_forest_train],

    'Model Accuracy Score': [
        log_accuracy, svc_accuracy, random_forest_accuracy
    ]
})



f_imp=pd.Series(random_forest.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print(f_imp)













