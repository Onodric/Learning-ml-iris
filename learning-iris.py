# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
# This is the pathe to the data
url = "./data/iris.data"
# These are the column names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Shape of the dataset
print(dataset.shape)

# Peaking at the head
# first 20 rows..
# note this is 0-indexed
# note also there are no units (which won't matter to the machine, but will matter to the user.
print(dataset.head(20))

# descriptions
# This gives some statistical data
# note also there are no units (which won't matter to the machine, but will matter to the user.
print(dataset.describe())

# class distribution
# note also there are no units (which won't matter to the machine, but will matter to the user.
# in this case, its the number of lines in each class
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# histograms
# this till show possible gaussian relationhips, which can be exploited by built-ins
dataset.hist()
# plt.show()

# scatter plot matrix
scatter_matrix(dataset)
# plt.show()

# up to this point, we are just looking at the data for possible analysis paths.
# Now we get cracking on the actual learning!

# Split-out validation dataset, 80% to learn, 20% for prediction testing
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = \
    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# print(X_train)
# print(X_validation)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
# Now lets build some models, linear and non-linear
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
# This shows that SVM is most accurate, but some say LDA and the tutorial says KNN
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "{}: {} ({})".format(name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
# plt.boxplot(results)
ax.set_xticklabels(names)
# plt.show()

# Make predictions on validation dataset
# using KNN
# KNN is only 90% confident
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("For KNN")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset
# using LDA
# LDA is indeed the most robust, at 97%
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print("For LDA")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset
# using SVM
# SVM is only 94%
svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print("For SVC")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


