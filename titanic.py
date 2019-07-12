import pandas as pd
import numpy as np
##import visuals as vs
from sklearn import tree
from sklearn import preprocessing
import graphviz
from sklearn.tree.export import export_text




data = pd.read_csv('titanic_data.csv')

print(data.columns)

y = data['Survived']
x = data.drop(['PassengerId', 'Survived', 'Pclass', 'Name','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

le = preprocessing.LabelEncoder()
x = x.apply(le.fit_transform)

y = np.array(y)
print(y)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)
pr= clf.predict(x)

r = export_text(clf)
print(r)

def accurecy_score(pred,y):
    return np.mean(y==pred)
print("acc"+str(accurecy_score(pr,y)))


def prediction0(data):
    pred = []

    for _,passenger in data.iterrows():
        if (passenger['Sex'] == 'female' and passenger['SibSp'] <3) or(passenger['Sex'] == 'male' and passenger['Age'] <=10):
             pred.append(1)
        else:
            pred.append(0)

    return pd.Series(pred)

prediction = prediction0(x)

acc = accurecy_score(prediction,y)



##vs.survival_stats(x, y, 'Pclass',["Sex == 'female'"])