import csv
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use("bmh")
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF




data = pd.read_csv('fifa_players.csv')
modified_data = data.drop('nationality_name', inplace=False, axis=1)

#Potential Datasets based on position...

#GOALKEEPERS
#goalkeeper_data = pd.read_csv('fifa_players.csv', usecols=['wage_eur', 'age', 'height_cm', 'weight_kg', 'nationality_name', 'overall', 'potential', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes'])
#goalkeeper_data1 = goalkeeper_data.drop(goalkeeper_data[goalkeeper_data['goalkeeping_reflexes'] < 40].index, inplace=False)
goalkeeper_data = modified_data.drop(modified_data[modified_data['goalkeeping_reflexes'] < 40].index, inplace=False)

#DEFENDERS
defender_data = modified_data.drop(modified_data[modified_data['defending_standing_tackle'] < 50].index, inplace=False)

#MIDFIELDERS


#ATTACKERS
attacker_data = modified_data.drop(modified_data[modified_data['attacking_finishing'] < 70].index, inplace=False)


'''
sns.set(rc={'figure.figsize':(15,15)})
corr = defender_data.corr()
sns.heatmap(corr[((corr >= 0) | (corr <= 0)) & (corr != 1)], annot=False, linewidths=.5, fmt='.2f')
plt.title('Corelation Matrix')
plt.show()
'''

y = modified_data['wage_eur']
X = modified_data.drop('wage_eur', inplace=False, axis=1)



scaler = preprocessing.StandardScaler()

scaledX = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(scaledX, y, test_size=0.2, random_state=15)



'''
#Hyper Parameter Grid Search
parameters = [{
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'C' : [1, 2, 3, 300, 500],
    'max_iter' : [1000, 100000]
}]


clf = GridSearchCV(SVC(), parameters, scoring='accuracy')
clf.fit(X_train, y_train)

print(clf.best_params_)
'''
#{'C': 3, 'kernel': 'rbf', 'max_iter': 100000}

kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X,y)
print(gpc.score(X,y))