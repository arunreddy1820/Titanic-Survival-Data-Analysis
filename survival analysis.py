import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='whitegrid')

# print(os.path.exists('D:\\21 projects\day-1\\21-Days-21-Projects-Dataset\Datasets/Titanic-Dataset.csv'))

titanic_df = pd.read_csv('D:\\21 projects\\day-1\\Titanic-Dataset.csv')
# print("First five rows:")

# print(titanic_df.head())

# print(titanic_df.tail())

# print(titanic_df.shape)
# print(titanic_df.info())

# print(titanic_df.describe())

# print(titanic_df['Cabin'].value_counts())

# print(titanic_df.isna().sum())

median_age = titanic_df['Age'].median()
# print(median)

titanic_df['Age'] = titanic_df['Age'].fillna(median_age)

# print("Missing values after cleaning")
# print(titanic_df.isna().sum())

# print(titanic_df[['Age', 'Embarked', 'Cabin']].isna().sum())

embarked_mode = titanic_df['Embarked'].mode()[0]
# print(embarked_mode)

# print(titanic_df['Embarked'])

titanic_df['Embarked'] = titanic_df['Embarked'].fillna(embarked_mode)

# print(titanic_df[['Age', 'Embarked', 'Cabin']].isna().sum())

titanic_df['Has_Cabin'] = titanic_df['Cabin'].notna().astype(int)

# print(titanic_df.info())

# print(titanic_df.head())

titanic_df.drop('Cabin', axis=1, inplace=True)

# print(titanic_df.info())
# print(titanic_df.describe())

# print(titanic_df.isna().sum())

print("Analyzing categorical features:")

fig,axes = plt.subplots(2, 3 ,figsize=(18, 12))
fig.suptitle('Univariate Analysis of Categorical Features', fontsize=16)

sns.countplot(ax=axes[0, 0], x = 'Survived', data = titanic_df).set_title('Survival Distribution')
sns.countplot(ax=axes[0, 1], x= 'Pclass', data=titanic_df).set_title("Passenger class distribution")
sns.countplot(ax = axes[0, 2], x= 'Sex', data=titanic_df).set_title("Gender Distribution")
sns.countplot(ax = axes[1, 0], x = 'Embarked', data = titanic_df).set_title("Port of Embarkation")
sns.countplot(ax = axes[1, 1], x = 'SibSp', data = titanic_df).set_title("Siblings/ Spouses Aboard")
sns.countplot(ax = axes[1, 2], x = 'Parch', data = titanic_df).set_title("Parents/ Children Aboard")



plt.tight_layout(rect=[0, 1, 1, 0.96])
plt.subplots_adjust(hspace=0.4)
# plt.show()


print("Analysizing Numerical Features")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Univariate Analysis of Numerical Features', fontsize=16)

sns.histplot(ax= axes[0], x = 'Age', data = titanic_df, kde=True, bins = 30).set_title("Age Distribution")
sns.histplot(ax = axes[1], x = 'Fare', data= titanic_df, kde=True, bins = 40).set_title("Fare Distribution")

# plt.show()



print("Bivariate Analysis: Feature vs Survival")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Bivariate Analysis with Survival', fontsize= 16)

# Pclass vs Survival
sns.barplot(ax = axes[0, 0], x = 'Pclass', y = 'Survived', data = titanic_df).set_title('Survival Rate by Pclass')

# Sex vs Survival
sns.barplot(ax = axes[0, 1], x = 'Sex', y='Survived', data = titanic_df).set_title("Survival Rate by Sex")

# Embarked vs Survival
sns.barplot(ax = axes[1, 0], x ='Embarked', y = 'Survived', data = titanic_df).set_title("Survival Rate by Embarked")

#has_cabin vs Survival
sns.barplot(ax = axes[1, 1], x= 'Has_Cabin', y = 'Survived', data = titanic_df).set_title("Survival Rate by Has_Cabin")

plt.tight_layout(rect = [0, 1, 1, 0.96])
plt.subplots_adjust(hspace=0.4)
# # plt.show()



#Age vs Survival
g = sns.FacetGrid(titanic_df, col='Survived', height=6)
g.map(sns.histplot, 'Age', bins = 25, kde=True)

g.fig.suptitle("Age Distribution by Survival Status", fontsize=16)
# plt.tight_layout(rect = [0, 1, 1, 0.96])
g.fig.subplots_adjust(top=0.9)
# plt.show()




plt.figure(figsize=(10, 8))
sns.boxplot(y='Fare', data=titanic_df)
plt.title('Box Plot of Ticket Fare')
plt.ylabel('Fare')
# plt.show()


titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

titanic_df['IsAlone'] = 0
titanic_df.loc[titanic_df['FamilySize'] == 1, 'IsAlone'] = 1
# print(titanic_df[['FamilySize', 'IsAlone']].head())


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(ax = axes[0], x = 'FamilySize', y='Survived', data= titanic_df).set_title('Survival Rate by Family Size')

sns.barplot(ax = axes[1], x= 'IsAlone', y='Survived', data = titanic_df).set_title("Survival rate for Those Travelling Alone")
# plt.show()



titanic_df['Title'] = titanic_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand = False)
# print(titanic_df['Title'].value_counts())

titanic_df['Title'] = titanic_df['Title'].replace(['Lady', 'Countess', 'Rev', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
titanic_df['Title'] = titanic_df['Title'].replace('Mlle', 'Miss')
titanic_df['Title'] = titanic_df['Title'].replace('Ms', 'Miss')
titanic_df['Title'] = titanic_df['Title'].replace('Mme', 'Mrs')

plt.figure(figsize=(12,6))
sns.barplot(x = 'Title', y='Survived', data=titanic_df)
plt.title('Survival Rate by Title')
plt.ylabel('Survival Proability')
# plt.show()



#Survival Rate by Pclass and Sex
sns.catplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df, kind='bar', height=6, aspect=1.5)
plt.title('Survival Rate by Pclass and Sex')
plt.ylabel('Survival Probability')
# plt.show()


plt.figure(figsize=(14, 8))
sns.violinplot(x='Sex', y='Age', hue='Survived', data=titanic_df, split=True, palette={0: 'blue', 1: 'orange'})
plt.title('Age Distribution by Sex and Survival')
# plt.show()


# Correlation Heatmap for numerical features
plt.figure(figsize=(14, 10))
numeric_cols = titanic_df.select_dtypes(include=np.number)
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.yticks(rotation=0)
plt.yticks(rotation=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()