import pandas as pd
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('titanic.csv', index_col='PassengerId')

featers = data.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

featers = featers.dropna()

featers['Sex'] = featers['Sex'].map({'female': 1, 'male': 0})
y = featers.Survived
featers = featers.drop('Survived', 1)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(featers, y)
imp = clf.feature_importances_

print featers
print imp
