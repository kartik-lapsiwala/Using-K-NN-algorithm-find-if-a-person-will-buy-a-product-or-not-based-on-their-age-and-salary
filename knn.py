# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# set up dependent and independent variables
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

# Divide dataset into training and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

nominator = cm[0][0]+cm[1][1]
denominator = nominator + cm[1][0] + cm[0][1]
accuracy = (nominator)/(denominator)*100

print('Accuracy of the model is {:.2f}%'.format(accuracy))

# Visualization
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01 ),
                    np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01 ))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T ).reshape(x1.shape), alpha = 0.45, 
             cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()