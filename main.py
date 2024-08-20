import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
print(df.head())


input("\n Press Enter to continue.\n")

#Data Cleaning
#Label encode the dataset
# df = util.labelEncoder(df, ["HeartDisease","GenHealth"])
df = util.labelEncoder(df, ["HeartDisease","Smoking","AlcoholDrinking","AgeCategory","PhysicalActivity","GenHealth","Sex"])


print("\nHere is a preview of the dataset after label encoding. \n")
print(df.head())


input("\nPress Enter to continue.\n")

#One hot encode the dataset
df = util.oneHotEncoder(df,["Race"]) 

print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head())


input("\nPress Enter to continue.\n")


#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split
#X holds the features, this is the list without heart disease data
X= df.drop("HeartDisease",axis = 1)

#y holds the target class aka the answer we want the model to give
y= df["HeartDisease"]

#assign multiple variables at once
# y represents target values
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=77)



from sklearn.tree import DecisionTreeClassifier
# creates a decision tree
clf = DecisionTreeClassifier(max_depth = 8, class_weight = "balanced")
#gives model X and y to train with
clf = clf.fit(X_train, y_train)


#Test the model with the testing data set and prints accuracy score
test_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
# now checking predictions and actual answers
test_acc = accuracy_score(y_test,test_predictions)

print("The accuracy with the testing data set of the Decision Tree is : " + str(test_acc))



#Prints the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,test_predictions, labels = [1,0])
#in a confusion matrix the first row represents patients with heart disease
#second row represents people without heart disease
#first column represents correct classification
#second column shows incorrect classification
print("The confusion matrix of the tree is : ")
print(cm)




#Test the model with the training data set and prints accuracy score
train_predictions = clf.predict(X_train)

from sklearn.metrics import accuracy_score
# now checking predictions and actual answers
train_acc = accuracy_score(y_train,train_predictions)

print("The accuracy with the training data set of the Decision Tree is : " + str(train_acc))




input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations
print("Decision trees are good for a field like books since users can enter their book preferences which the decision tree uses as its splitting nodes. The more specifications the user gives the better the recommended book which cuts down all the hassle of searching for the perfect book.\n")
print("Some factors that I will have to be careful to ensure that my model performs fairly are : data bias, under-representation of a demographic and preventing overfitting of data.")






#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

util.printTree(clf, X.columns)