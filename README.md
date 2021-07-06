# Hyperparameter Tuning
A **Streamlit** app for hyperparameter tuning of classifier models using GridSearchCV
- https://share.streamlit.io/randell-janus/hyperparameter-tuning/main/app.py
- Deployed via [Streamlit Sharing](https://streamlit.io/sharing)

## About
### GridSearchCV
- A function that comes from the Scikit-learn's [model_selection package](https://scikit-learn.org/stable/model_selection.html).
- It helps to loop through predefined hyperparameters and fit your estimator (model) on your training set.

### Dataset Source  
The datasets used are the Iris Plants and Wine Recognition from Scikit-learn's [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html).

### Models Used
* Random Forest Classifier
* Support Vector Machine
* Logistic Regression
  
## Web App Features  
- Switch between two datasets
- Configurable Parameters
  - Random Forest - tune the n_estimators and max_depth.
  - SVM - Specify the kernel types to be included in the tuning and tune C parameter.
  - Logistic Regression - Specify the norm used in the penalization and customize the C parameter.

## Views
- Main page ![](https://github.com/Randell-janus/hyperparameter-tuning/blob/main/public/home.JPG)
- Full user settings view & Parameter outputs ![](https://github.com/Randell-janus/hyperparameter-tuning/blob/main/public/home-ext.JPG)