import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in dataframe. Shape is (8274, 29)
df = pd.read_csv('astros_bangs_20200127.csv')

# Define target as whether bangs were heard during batter's AB
# Bangs were used by the Astros after using technology to steal signs to signal to the batter what pitch was coming
y = 'has_bangs'

# Feature engineering function
def feature_engineering(X):
  '''
  Engineer new features for dataframe X.
  '''
  hit_list = ['Single', 'Double', 'Triple', 'Home Run']
  obp_list = ['Single', 'Double', 'Triple', 'Home Run', 'Walk', 'Hit By Pitch', 'Sac Fly']
  out_list = ['Strikeout', 'Double Play',
       'Grounded Into DP', 'Catcher Interference', 'Flyout', 'Groundout',
       'Lineout', 'Sac Fly', 'Fan interference', 'Pop Out',
       'Double', 'Forceout', 'Sac Bunt', 'Strikeout - DP', 'Fielders Choice Out',
       'Bunt Groundout', 'Bunt Pop Out']
  X['hit'] = X['at_bat_event'].isin(hit_list)
  X['on_base'] = X['at_bat_event'].isin(obp_list)
                                          
  return X

df = feature_engineering(df)

def wrangle(X):
    '''
    Wrangle function for DF before train_test_split
    '''

    X = X.copy()

    # Change game_date to a datetime for ease of use
    X['game_date'] = pd.to_datetime(X['game_date'], infer_datetime_format=True)

    # Replace some character strings with numbers
    X['has_bangs'] = X['has_bangs'].replace({'n': 0, 'y': 1})
    X['on_1b'] = X['on_1b'].replace({'t': 1, 'f': 0})
    X['on_2b'] = X['on_2b'].replace({'t': 1, 'f': 0})
    X['on_3b'] = X['on_3b'].replace({'t': 1, 'f': 0})
    X['hit'] = X['hit'].replace({True: 1, False: 0})
    X['on_base'] = X['on_base'].replace({True: 1, False: 0})

    # Drop these columns since they are video playback related or contain no cardinality
    X = X.drop(columns=['youtube_id', 'pitch_youtube_seconds', 'youtube_url', 
                    'pitch_datetime', 'game_pitch_id', 'event_number', 
                    'pitch_playid', 'atbat_playid', 'bangs'])

    # Save only data where bangs occured (95% of df), remove the rest
    games_with_bangs_list = X[(X[y]==1)].game_date.to_list()
    X = X[X['game_date'].isin(games_with_bangs_list)]

    # Remove rows with runners getting out to end the inning (<.2% of outcomes)
    # Since it will conflict with any batting average and obp calculations
    no_interference = ['Home Run', 'Single', 'Strikeout', 'Double Play', 
                   'Grounded Into DP', 'Flyout', 'Groundout', 'Lineout', 'Walk',
                   'Sac Fly', 'Pop Out', 'Double', 'Forceout', 'Field Error', 
                   'Sac Bunt', 'Hit By Pitch', 'Strikeout - DP', 'Triple', 
                   'Fielders Choice Out', 'Bunt Groundout', 'Bunt Pop Out']
    X = X[X['at_bat_event'].isin(no_interference)]

    # Drop columns that may contain data leakage / that aren't useful in predicting real time (final runs)
    X = X.drop(['game_date', 'final_away_runs', 'final_home_runs'], axis=1)

    return X

df = wrangle(df)

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve

# create a features DF without the target
X = df.drop(y, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, df[y], test_size=0.2, random_state=17)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=17) # 0.25 x 0.8 = 0.2\

def run_model(X, y, X_train, X_val, X_test, y_train, y_val, y_test):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    import category_encoders as ce
    from sklearn.feature_selection import SelectKBest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
    from sklearn.metrics import roc_curve
    pipeline = make_pipeline(
        SimpleImputer(),
        ce.OrdinalEncoder(),
        SelectKBest(),
        RandomForestClassifier(n_jobs=-1, n_estimators=24),
)

# Use this for actual hyperparameter tuning
# param_distributions = {
#     'randomforestclassifier__n_estimators': [25],
#     'randomforestclassifier__max_depth': [None, 30, 32, 34, 36],
#     'randomforestclassifier__min_samples_split': [6, 8, 10, 12, 14],
#     'randomforestclassifier__criterion': ['gini', 'entropy'],
#     'randomforestclassifier__min_samples_leaf': [1, 2],
#     'randomforestclassifier__max_leaf_nodes': [None],
#     'randomforestclassifier__oob_score': [True, False],
#     'selectkbest__k': [10, 15, 20, 25, 30],
#     'simpleimputer__strategy': ['most_frequent'],
# }

# Use this for final model (one parameter each)
    param_distributions = {
        'randomforestclassifier__n_estimators': [25],
        'randomforestclassifier__max_depth': [32],
        'randomforestclassifier__min_samples_split': [10],
        'randomforestclassifier__criterion': ['entropy'],
        'randomforestclassifier__min_samples_leaf': [2],
        'randomforestclassifier__max_leaf_nodes': [None],
        'randomforestclassifier__oob_score': [True],
        'selectkbest__k': [10],
        'simpleimputer__strategy': ['most_frequent'],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=1,
        cv=5,
        scoring='accuracy',
        verbose=0,
        return_train_score=True,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    pipeline = search.best_estimator_
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    pipeline.score(X_test, y_test)
    y_pred_prob = pipeline.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)

    pd.DataFrame({
        'False Positive Rate': fpr, 
        'True Positive Rate': tpr, 
        'Threshold': thresholds
    })

    # Figure: ROC_AUC_graph
    plt.scatter(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_val, y_pred_prob)

    print('Best hyperparameters: ', search.best_params_)
    print('Cross-validation Accuracy: ', search.best_score_)
    print('ROC_AUC_Score', roc_auc_score(y_val, y_pred_prob))

    # Get feature importances
    rf = pipeline.named_steps['randomforestclassifier']
    importances = rf.feature_importances_

    # define standard deviation for graph
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print('indices:', indices)
    # Print the feature rank
    # for f in range(X.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]));
    print(X.head())
    print('X.shape[1]:', X.shape[1])
    # Plot feature importances
    plt.figure()
    plt.title("Feature Importances")
    plt.barh(range(X.shape[1]), importances[indices], color='grey', yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    return search.best_score_

model_score = run_model(X, y, X_train, X_val, X_test, y_train, y_val, y_test)

print('Model Validation Accuracy: ', model_score)
'''
# Check which features were selected
included_mask = pipeline['selectkbest'].get_support()
all_names = X_train.columns
included_names = all_names[included_mask]

# the tilde operator returns included_mask's compliment
excluded_names = all_names[~included_mask]

print('Included Features: ')
for name in included_names:
    print(name)
print('\n')
print('Excluded Features: ')
for name in excluded_names:
    print(name)
'''
# Best hyperparameters:  {'simpleimputer__strategy': 'most_frequent', 'selectkbest__k': 10, 'randomforestclassifier__oob_score': True, 'randomforestclassifier__n_estimators': 25, 'randomforestclassifier__min_samples_split': 10, 'randomforestclassifier__min_samples_leaf': 2, 'randomforestclassifier__max_leaf_nodes': None, 'randomforestclassifier__max_depth': 32, 'randomforestclassifier__criterion': 'entropy'}
# Cross-validation Accuracy:  0.9155555555555556

# mt