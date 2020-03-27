# Machine Learning Modeler Tool

Create machine learning model from csv file using sklearn API

## Modeler

$python mlmodeler.py m csv_file y_column method['knn','tree','randomf','svc'] 0

### Example:

$python mlmodeler.py m wine_dataset.csv "Wine Type" knn 0

## Predict

$python mlmodeler p csv_file model_file.joblib

### Example:

$python mlmodeler.py p wine_dataset_train.csv wine_dataset_knn.joblib

