from sklearn.datasets import fetch_openml
%matplotlib inline
import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LogisticRegression

#Fetching dataset
mnist_data=fetch_openml('mnist_784')

x=mnist_data.data
y=mnist_data.target

#plotting a random number

random_num = x[36001]
image = random_num.reshape(28, 28)
plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")

x_train,x_test=x[:60000],x[60000:]
y_train,y_test=y[:60000],y[60000:]

shuffel_index = np.random.permutation(600)
x_train = x_train[shuffel_index]
y_train = y_train[shuffel_index]
y_train3 = (y_train=='3')
y_test3 = (y_test=='3')

lrmodel = LogisticRegression()
lrmodel.fit(x_train, y_train3)
lrmodel.predict([random_num])

accuracy = cross_val_score(lrmodel, x_train, y_train3, cv=3, scoring="accuracy")
accuracy = accuracy.mean()
accuracy = accuracy*100
print(accuracy)