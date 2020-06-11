# Importing Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, f1_score, classification_report

# Importing Data
titanic = pd.read_csv('train.csv')
pd.set_option('display.max_columns',len(titanic.columns))

print('Summary Statistics:')
print(titanic.head())
# Summary Data for Numeric Columns
print(titanic.describe())
# Summary Data for all columns
print(titanic.describe(include='all'))

def prep_data(df):
    """Reusable function to clear train and test data"""

    Embarked_dict = {
        'S':0,
        'C':1,
        'Q':2,
    }

    df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)

    # Fill Embarked column with average value
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map(Embarked_dict, na_action='ignore').astype(int)

    # temporary fare_group measure
    df['Fare_group'] = 0
    def fare_group(x):
        if int(x) in range(0, 75):
            return 0
        elif int(x) in range(75, 100):
            return 1
        elif int(x) in range(100, 200):
            return 2
        else:
            return 3

    df['Fare'] = df['Fare'].fillna(df.Fare.median())
    df['Fare_group'] = df['Fare'].map(fare_group).astype(int)
    print(df.Fare_group)

    # Code from Kaggle Notebook -> creating ordinal title column
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

    # Code from Kaggle Notebook -> creating 'FamilySize' and then 'IsAlone'
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Drop columns
    """
    We can drop Cabin and Ticket because we're not using them, and Name because we've extracted Nulls.
    """

    df.drop(['Cabin', 'Ticket', 'Name'], axis = 1, inplace = True)


prep_data(titanic)

# Drop remaining NA's from training data
titanic.dropna(inplace = True)

print(titanic.head())

# Generate Histogram Plots (Distributions of Each Numeric Column)
for i in titanic.columns:
    if titanic[i].dtypes == 'int64' or titanic[i].dtypes == 'float64':
        sns.distplot(titanic[i])
        plt.xlabel(i)
        plt.ylabel('Frequency')
        plt.title(str('Distribution of ' + i))
        plt.show()


multigrid = sns.FacetGrid(titanic, col= 'Survived')
multigrid.map(plt.hist, 'Embarked', bins = 3)
plt.show()

next_data = titanic[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot(next_data['Pclass'], next_data['Survived'])
plt.xlabel('Passenger Class')
plt.xticks([0,1,2], labels = ['First Class', 'Second Class', 'Third Class'])
plt.ylabel('Survival Rate')
plt.title(str('Average Survival Rate by Passenger Class'))
plt.show()


next_data = titanic[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(next_data['Embarked'], next_data['Survived'])
plt.xlabel('Embarked Location')
plt.ylabel('Survival Rate')
plt.title(str('Average Survival Rate by Embarkment Location'))
plt.show()

# Generating correlation matrix
correlation = titanic.corr()
sns.heatmap(correlation, annot=True)
plt.title("Titanic Heatmap")
plt.show()

# Selecting Variables
features = ['Pclass', 'Fare', 'Sex', 'Embarked']
X = titanic[features]
y = titanic.Survived

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Initialize Random Forest Classifier
titanic_model = RandomForestClassifier(random_state = 1)

# Fit model to training data
titanic_model.fit(train_X, train_y)

# predictions
predictions = titanic_model.predict(val_X)

# Scoring Predictions
mae = mean_absolute_error(val_y, predictions)
report = classification_report(val_y, predictions,target_names=['Not Survived', 'Survived'])

#score = accuracy_score(val_y, predictions.round(), normalize = False)
print('Mean absolute error is', mae)
print('Classification report: \n', report)

if True:
    # If true, generate file for Kaggle
    titanic_test = pd.read_csv('test.csv')
    print(titanic_test.shape)
    prep_data(titanic_test)
    test_X = titanic_test[features]
    test_X.fillna(test_X.median(), inplace =True)
    test_predictions = titanic_model.predict(test_X)
    print(test_X.shape, test_predictions.shape)
    output = pd.DataFrame({'PassengerId': titanic_test.PassengerId,
                           'Survived': test_predictions})
    output.to_csv('submission.csv', index=False)