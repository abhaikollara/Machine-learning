import pandas as pd
import numpy as np
import random
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# from sklearn.
def selectFeatures(data):
    
        
    #Add new feautres
    data['Family'] = data['SibSp'] + data['Parch'] #Total family members on board
    data.loc[data['Family'] > 0, 'Family'] = 1
    data.loc[data['Family'] == 0, 'Family'] = 0

    def get_person(passenger):
        age,sex = passenger
        if age < 15:
            return 'child'
        else:
            return sex
    pclassDummies  = pd.get_dummies(data['Pclass'])
    pclassDummies.columns = ['Class_1','Class_2','Class_3']
    pclassDummies.drop(['Class_3'], axis=1, inplace=True)
    data = data.join(pclassDummies)

    data['Person'] = data[['Age','Sex']].apply(get_person,axis=1)

    # PersonDummies  = pd.get_dummies(data['Person'])
    # PersonDummies.drop(['male'], axis=1, inplace=True)


    embarkDummies = pd.get_dummies(data['Embarked'])
    embarkDummies.drop(['S'], axis=1, inplace=True)
    data = data.join(embarkDummies)

    #Remove useless features
    data.drop(['PassengerId','Name','Ticket','Cabin','Sex','SibSp','Person','Embarked','Parch','Fare'],axis=1,inplace=True)
    
    #Fill unavailabe data
    data['Age'] = data['Age'].fillna(random.randint(10,60))
    data['Age'] = data['Age'].astype(int)

    return data

def seperate(data):
    #Seperate features and labels
    data_features = data.drop(['Survived'],1)
    data_labels = data['Survived']

    return data_features,data_labels

##TEST DATA PREDICTION
def submit():
    testData = pd.read_csv("../datasets/Titanictest.csv")
    PID = testData['PassengerId']
    testData = selectFeatures(testData)

    predictions = model.predict(testData)
    submission = pd.DataFrame({ "PassengerID" : PID, "Survived" : predictions })

    submission.to_csv("results.csv",index=False)

##############################################################3
titanic = pd.read_csv("../datasets/Titanictrain.csv")
data = selectFeatures(titanic)
trainFeatures, trainLabels = seperate(data)


#Fit and predict
# model = RandomForestClassifier(n_estimators=100)
model = XGBClassifier(n_estimators=100, learning_rate=0.1)
model.fit(trainFeatures, trainLabels, verbose=True)

accuracies = cross_validation.cross_val_score(model,trainFeatures,trainLabels,cv=6)

print ("Score : %.6f" % accuracies.mean())
submit()

plot_importance(model)
plt.show()