import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


names = ["height", "length", "area", "eccen", "p_black",
         "p_and", "mean_tr", "blackpix", "blackand", "wb_trans"]

# df = rawData.to_csv('data_out.csv', header=None, float_format='%.3f')

pageBlocksData = pd.read_csv('data_out.csv', names=names)


x = pageBlocksData.drop('wb_trans', axis=1)
y = pageBlocksData['wb_trans']

X_train, X_test, y_train, y_test = train_test_split(x, y)
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#  One hidden layer with 10 nodes on that layer
mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000)
mlp.fit(X_train, y_train)


predictions = mlp.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
