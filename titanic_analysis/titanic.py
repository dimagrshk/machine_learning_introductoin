import pandas as pd
import parse_name

#1
data = pd.read_csv('titanic.csv', index_col='PassengerId')
male, famale = data.Sex.value_counts()
print 'Mens and womens:', male, famale

#2
dead, survived = data.Survived.value_counts(1)
print 'How much percent of passangers survived: {0:.2f}.'.format(survived*100)

#3
third, first, second = data.Pclass.value_counts(1)
print 'How much passangers in fist class: {0:.2f}'.format(first*100)

#4
print 'Mean: {0:.2f}'.format(data.Age.mean()), 'and median: {0:.2f}'.format(data.Age.median())

#5
print 'Correlation: {0:.2f}'.format(data.Parch.corr(data.SibSp))

#6
df_famale = data[data.Sex == "female"]
frstName = df_famale.Name.apply(parse_name.simpleParse)
often = frstName.mode()
print often