import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
import warnings
import pickle
warnings.filterwarnings("ignore")

seed = 42

df = pd.read_csv("features_data.csv")
data = np.array(df)

df_features = df[['birth_year', 'country', 'city', 'plan', 'num_contacts',
       'User_gain_usd', 'Transaction_per_user',
       'notification_received', 'target']]
# initializing encoders for each column
le_target = preprocessing.LabelEncoder()
le_country = preprocessing.LabelEncoder() 
le_city = preprocessing.LabelEncoder() 
le_plan = preprocessing.LabelEncoder() 
# getting the labels in columns  
df_features["target"] = le_target.fit_transform(df_features["target"])
df_features["country"] = le_country.fit_transform(df_features["country"])
df_features["city"] = le_city.fit_transform(df_features["city"])
df_features["plan"] = le_plan.fit_transform(df_features["plan"])

y = df_features.target
X = df_features.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=seed)
# Create the Decision Tree classifier with the optimal hyperparameters as found by GridSearchCV
tree_clf = Pipeline([
    ('select_features', SelectKBest(k=7)),
    ('classify', DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=None, min_samples_split=20))
])

tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)
score = tree_clf.score(X_test, y_test)
print (score)

with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(score) + "\n")


# Plot it
disp = plot_confusion_matrix(tree_clf, X_test, y_test, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')

pickle.dump(tree_clf,open('my_model.pkl','wb'))
