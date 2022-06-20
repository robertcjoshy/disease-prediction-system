from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# reading trained data-------------
df = pd.read_csv("training.csv")
df
target = df.prognosis
inputs = df.drop('prognosis',axis='columns')

# accuracy_score -------------------------------
#test_x = pd.read_csv('testing.csv')

# naive bayes------------------------
def naivebayes(n):
    rober = MultinomialNB()
    rober.fit(inputs,target)
    list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    joshy = rober.predict(n)
    
    daisy = rober.predict_proba(n)
    max = daisy[0][0]
    for i in range(0,40):
        if daisy[0][i] > max:
            max = daisy[0][i]
    res = round(max,2)
    result = res * 100
    return joshy, result

# decision tree--------------------------
def decisiontree(n):
    robe = tree.DecisionTreeClassifier()
    robe.fit(inputs,target)
    robe.score(inputs,target)
    list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    joshy = robe.predict(n)
    daisy = robe.predict_proba(n)
    max = daisy[0][0]
    for i in range(0,40):
        if daisy[0][i] > max:
            max = daisy[0][i]
    res = round(max,2)
    result = res * 100
    return joshy, result   

# random forest--------------------------------
def randomforest(n):
    rob = RandomForestClassifier()
    rob.fit(inputs,target)
    list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    joshy = rob.predict(n)
    daisy = rob.predict_proba(n)
    max = daisy[0][0]
    for i in range(0,40):
        if daisy[0][i] > max:
            max = daisy[0][i]
    res = round(max,2)
    result = res * 100
    return joshy, result 



