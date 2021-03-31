import streamlit as st 
import pandas as pd
import numpy as np 

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

st.set_page_config(page_title='Hyperparameter Tuning', layout='wide')


st.title('Hyperparameter Tuning with Scikit-Learn!')
st.write("Tune **parameters** with Scikit-Learn's *GridSearchCV* ")
st.write('')

col1, col2 =st.beta_columns(2)
about_expander = col1.beta_expander('About',expanded=False)
with about_expander:
    st.info("""
             This web application is a simple demonstration of Hyperparameter tuning with 
             **GridSearchCV**. The parameters customizable in this app are only limited 
             and the algorithms and datasets used are from Scikit learn. There may be other combinations 
             of parameters and algorithms that can yield a better accuracy score for a given dataset.
             """)
info_expander = col2.beta_expander('What is Hyperparameter Tuning?',expanded=False)
with info_expander:
    st.info("""
             **Hyperparameters** are the parameters that describe the model architecture and 
             **hyperparameter tuning** is the method of looking for the optimal model architecture
             """)
st.sidebar.header('Select Dataset')

dataset_name = st.sidebar.selectbox('', ('Iris Plants', 'Wine Recognition'))
st.title('')
st.write(f"## **{dataset_name} Dataset**")
classifier = st.sidebar.selectbox('Select Classifier', 
                                  ('Random Forest', 'SVM', 'Logistic Regression'))
cv_count = st.sidebar.slider('Cross-validation count', 2, 5, 3)
st.sidebar.write('---')
st.sidebar.subheader('User Input Parameters')
st.sidebar.write('')

def get_dataset(name):
    df = None
    if name == 'Iris Plants':
        df = datasets.load_iris()
    elif name == 'Wine Recognition':
        df = datasets.load_wine()
    X = df.data
    y = df.target
    return X, y

X, y = get_dataset(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def get_classifier(clf_name):
    model = None
    parameters = None
    
    if clf_name == 'SVM':
        st.sidebar.write("**Kernel Type**")
        st.sidebar.write('Specifies the kernel type to be used in the algorithm.')
        kernel_type = st.sidebar.multiselect('', options=['linear', 'rbf', 'poly'], default=['linear', 'rbf', 'poly'])
        st.sidebar.subheader('')
        
        st.sidebar.write('**Regularization Parameter**')
        st.sidebar.write('The strength of the regularization is inversely proportional to C.')
        c1 = st.sidebar.slider('C1', 1, 7, 1)
        c2 = st.sidebar.slider('C2', 8, 14, 10)
        c3 = st.sidebar.slider('C3', 15, 20, 20)
        
        parameters = {'C':[c1, c2, c3], 'kernel':kernel_type}
        model = svm.SVC()
    elif clf_name == 'Random Forest':
        st.sidebar.write('**Number of Estimators**')
        st.sidebar.write('The number of trees in the forest.')
        n1 = st.sidebar.slider('n_estimators1', 1, 40, 5)
        n2 = st.sidebar.slider('n_estimators2', 41, 80, 50)
        n3 = st.sidebar.slider('n_estimators3', 81, 120, 100)
        st.sidebar.header('')
        
        st.sidebar.write('**Max depth**')
        st.sidebar.write('The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.')
        md1 = st.sidebar.slider('max_depth1', 1, 7, 1)
        md2 = st.sidebar.slider('max_depth2', 8, 14, 10)
        md3 = st.sidebar.slider('max_depth3', 15, 20, 20)
        
        parameters = {'n_estimators':[n1, n2, n3], 'max_depth':[md1, md2, md3]}
        model = RandomForestClassifier()
    else:
        st.sidebar.write("**Penalty**")
        st.sidebar.write('Used to specify the norm used in the penalization.')
        penalty = st.sidebar.multiselect('', options=['l1', 'l2'], default=['l1', 'l2'])
        st.sidebar.subheader('')
        
        st.sidebar.write('**Regularization Parameter**')
        st.sidebar.write('Inverse of regularization strength; must be a positive float.')
        c1 = st.sidebar.slider('C1', 0.01, 1.00, 0.05)
        c2 = st.sidebar.slider('C2', 2, 19, 10)
        c3 = st.sidebar.slider('C3', 20, 100, 80, 10)
        
        parameters = {'penalty':penalty, 'C':[c1, c2, c3]}
        model = LogisticRegression(solver='liblinear', max_iter=200)
        
    return model, parameters

model, parameters = get_classifier(classifier)

clf = GridSearchCV(estimator=model, param_grid=parameters, cv=cv_count, return_train_score=False)
clf.fit(X, y)

df = pd.DataFrame(clf.cv_results_)

st.header('Tuning Results')
results_df = st.multiselect('', options=['mean_fit_time', 'std_fit_time', 'mean_score_time', 
                                         'std_score_time', 'split0_test_score', 'split1_test_score', 
                                         'split2_test_score', 'std_test_score', 'rank_test_score'], 
                            default=['mean_score_time', 'std_score_time', 
                                     'split0_test_score', 'split1_test_score', 
                                     'split2_test_score'])
df_results = df[results_df]
st.write(df_results)

st.subheader('**Parameters and Mean test score**')
st.write(df[['params', 'mean_test_score']])
st.write('Best Score:', clf.best_score_)
st.write('Best Parameters:', clf.best_params_)
