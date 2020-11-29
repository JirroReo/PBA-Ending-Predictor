import pandas as pd
import os
from sportsreference.ncaab.teams import Teams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

FIELDS_TO_DROP = ['away_points', 'home_points', 'date', 'location',
                  'losing_abbr', 'losing_name', 'winner', 'winning_abbr',
                  'winning_name', 'home_ranking', 'away_ranking']

dataset = pd.DataFrame()
teams = Teams()
print("Concatenating...\n")
a = 1
for team in teams:
    dataset = pd.concat([dataset, team.schedule.dataframe_extended])
    print(a)
    a+=1
X = dataset.drop(FIELDS_TO_DROP, 1).dropna().drop_duplicates()
y = dataset[['home_points', 'away_points']].values
print("Saving to CSV..\n")
df = dataset
df.to_csv(r'C:\Users\Jiro\Downloads\Spaghetti\NCAAB Dataset.csv', index=False)
print("Training the model...\n")
X_train, X_test, y_train, y_test = train_test_split(X, y)
parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}
model = RandomForestRegressor(**parameters)
model.fit(X_train, y_train)
print(model.predict(X_test).astype(int), y_test)