import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

datainput = pd.read_csv("salary.csv") # Read data from csv file

X = datainput[['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','hours-per-week','native-country']].values

# Data Preprocessing
from sklearn import preprocessing # changes non-numerical values to numbers so that algorithms can work on them

X[:, 0] = datainput['age']

label_workclass = preprocessing.LabelEncoder()
label_workclass.fit([' Federal-gov',' Local-gov', ' Private',' Self-emp-inc',' Self-emp-not-inc',' State-gov'])
X[:, 1] = label_workclass.transform(X[:, 1])

label_education = preprocessing.LabelEncoder()
label_education.fit([' 10th', ' 11th',' 12th',' 1st-4th',' 5th-6th',' 7th-8th',' 9th',' Assoc-acdm',' Assoc-voc',' Bachelors',' Doctorate',' HS-grad',' Masters',' Preschool',' Prof-school',' Some-college'])
X[:, 2] = label_education.transform(X[:, 2])

X[:, 3] = datainput['education-num']

label_marital_status = preprocessing.LabelEncoder()
label_marital_status.fit([' Divorced',' Married-AF-spouse',' Married-civ-spouse',' Married-spouse-absent',' Never-married',' Separated',' Widowed'])
X[:, 4] = label_marital_status.transform(X[:, 4])

label_occupation = preprocessing.LabelEncoder()
label_occupation.fit([' Adm-clerical',' Armed-Forces',' Craft-repair',' Exec-managerial',' Farming-fishing',' Handlers-cleaners',' Machine-op-inspct',' Other-service',' Priv-house-serv',' Prof-specialty',' Protective-serv',' Sales',' Tech-support',' Transport-moving'])
X[:, 5] = label_occupation.transform(X[:, 5])

label_relationship = preprocessing.LabelEncoder()
label_relationship.fit([' Husband',' Not-in-family',' Other-relative',' Own-child',' Unmarried',' Wife'])
X[:, 6] = label_relationship.transform(X[:, 6])

label_race = preprocessing.LabelEncoder()
label_race.fit([' Amer-Indian-Eskimo',' Asian-Pac-Islander',' Black',' Other',' White'])
X[:, 7] = label_race.transform(X[:, 7])

label_sex = preprocessing.LabelEncoder()
label_sex.fit([' Male',' Female'])
X[:, 8] = label_sex.transform(X[:, 8])

X[:, 9] = datainput['hours-per-week']

label_native_country = preprocessing.LabelEncoder()
label_native_country.fit([' Cambodia',' Canada',' China',' Columbia',' Cuba',' Dominican-Republic',' Ecuador',' El-Salvador',' England',' France',' Germany',' Guatemala',' Haiti',' Honduras',' India',' Iran',' Italy',' Jamaica',' Japan',' Laos',' Mexico',' Philippines',' Poland',' Portugal',' Puerto-Rico',' South',' Taiwan',' Thailand',' United-States',' Yugoslavia'])
X[:, 10] = label_native_country.transform(X[:, 10])

datainput['salary']=datainput.salary.replace(1,'<=50K')
datainput['salary']=datainput.salary.replace(0,'>50K')

y = datainput["salary"]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
h=RandomForestClassifier(n_estimators=100,random_state=1)

h.fit(x_train,y_train)
y_pred=h.predict(x_test)

print(pd.crosstab(y_test,y_pred))
print(classification_report(y_test,y_pred))



