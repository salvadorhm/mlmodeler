import shutil # Maneja archivos y directorios.
import cgi,os # Ejecuta un programa en el servidor y despliega su resultado hacia el cliente.
import cgitb # Proporciona un controlador especial para scripts de Python. 
import os, sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import dump, load

config = {}

def train_validation(csv_file):
    try:
        df = pd.read_csv(csv_file)
        train, validation = train_test_split(df, test_size=0.1)
        print("Creating train.csv")
        train.to_csv(config['file'] + "_"+"train.csv", sep=',',index=False)
        print("Creating validation.csv")
        validation.to_csv(config['file'] + "_"+"validation.csv", sep=',',index=False)
    except Exception as e:
        print(e.args[0])

def selectX():
    try:
        print("Selecting X...")
        dataframe = pd.read_csv(config['file'] + "_"+"train.csv")
        cols = list(dataframe)
        columns = []
        cols.remove(config['y'])
        for row in cols:
            if dataframe[row].dtypes != 'object' and dataframe[row].isnull().sum() == 0:
                columns.append((row))
        config['x'] = columns
    except Exception as e:
        print(e.args[0])

def selectX2():
    try:
        dataframe = pd.read_csv(config['csv_file'])
        cols = list(dataframe)
        columns = []
        # cols.remove(config['y'])
        for row in cols:
            if dataframe[row].dtypes != 'object' and dataframe[row].isnull().sum() == 0:
                columns.append((row))
        config['x'] = columns
    except Exception as e:
        print(e.args[0])

def df():
    try:
        dataframe = pd.read_csv(config['file'] + "_"+"train.csv")
        print("Dataframe x")
        df_x = dataframe[config['x']]
        print("Dataframe y")
        df_y = dataframe[config['y']]
        x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
        print("Dataframe x_train")
        config['x_train'] = x_train
        print("Dataframe x_test")
        config['x_test'] = x_test
        print("Dataframe y_train")
        config['y_train'] = y_train
        print("Dataframe y_test")
        config['y_test'] = y_test
    except Exception as e:
        print(e.args[0])

def train():
    try:
        if config['method'] == "knn":
            model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
        elif config['method'] == "tree":
            model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)
        elif config['method'] == "randomf":
            model = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=1, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
        elif config['method'] == "svm":
            model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
        else:
            model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)

        model.fit(config['x_train'],config['y_train'])
        dump(model, config['file'] + "_" + config['method']+".joblib") 
        # Predictions
        predictions = model.predict(config['x_test'])
        # Evaluation
        report = classification_report(config['y_test'], predictions)
        confusion = confusion_matrix(config['y_test'], predictions)
        score= model.score(config['x_test'],config['y_test'])
        # accuracy= model.accuracy_score(config['x_test'],config['y_test'])
        data_compare = pd.DataFrame({"Actual":config['y_test'], "Predicted":predictions})
        # Results
        print("Confusion matrix",confusion)
        print("Score: ",score)
        # print("Accuracy : ",accuracy)
        if config['verbose'] != '0':
            print(data_compare)
    except Exception as e:
        print(e.args[0])

def validate():
    try:
        model = load(config['model_file'])
        # dataframe_test = pd.read_csv(config['file'] + "_"+"validation.csv")
        dataframe_test = pd.read_csv(config['csv_file'])
        xs = dataframe_test[config['x']]
        ys = dataframe_test[config['y']]
        predictions = model.predict(xs)
        data_compare_test = pd.DataFrame({"Actual":ys, "Predicted":predictions})
        print(data_compare_test)
    except Exception as e:
        print(e.args[0])

def predict():
    try:
        model = load(config['model_file'])
        dataframe_test = pd.read_csv(config['csv_file'])
        xs = dataframe_test[config['x']]
        predictions = model.predict(xs)
        print(list(predictions))
    except Exception as e:
        print(e.args[0])

def main():
    try:
        args = sys.argv
        mode = str(args[1])  # Verbose
       
        if mode == 'm':
            csv_file = str(args[2])  # csv_file 
            y = str(args[3])  # Dependent term Y
            file = csv_file.split(".")[0]
            method = str(args[4])  # Method sklearn
            verbose = str(args[5])  # Verbose
            config['file'] = file
            config['y'] = y
            config['method'] = method
            config['mode'] = verbose
            train_validation(csv_file)
            selectX()
            df()
            train()
        elif mode == 'p':
            csv_file = str(args[2])  # csv_file 
            model_file = str(args[3])  # model sklearn
            config['csv_file'] = csv_file
            config['model_file'] = model_file
            selectX2()
            predict()
    except Exception as e:
        print(e.args[0])

main()
