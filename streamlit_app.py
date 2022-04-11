# import libraries
import streamlit as st  # For web application
import pandas as pd  # Python data analysis library, Name derived from panel data
import numpy as np  # For arithmatic operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For datasets
import sklearn  # For ml models
from sklearn import datasets
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Application title
st.markdown('''
# Streamlit Web Application for Classification Problem''') 

with st.sidebar:
    get_ds_name = st.sidebar.selectbox('Select Dataset',
                ('Iris','Breast Cancer', 'Wine'))  

    get_classifier_name = st.sidebar.selectbox('Select Classifier',
                ('KNN','SVM', 'Random Forest')) 


# import dataset selected by user


def get_dataset(dataset_name):
    
    global dataset
    # declare variable and assign none value
    

    if dataset_name == 'Iris':
        
        dataset = datasets.load_iris()
    
    elif dataset_name == 'Wine':
        
        dataset = datasets.load_wine()
    
    else:
        
        dataset = datasets.load_breast_cancer()
    
    
    # get inpt features from dataset
    X = dataset.data

    # get target variable from dataset
    y = dataset.target
    
    # return input and target
    return X, y

# Function calling
X, y = get_dataset(get_ds_name)

# check shape of dataset
st.write("Rows, Columns: ", X.shape)
st.write("Number of classes: ", len(np.unique(y))) # target variable is y and its type is class. unique values ki length lay lain gay.

# Define parameter of all classifiers. Here are three classifiers.
def get_model_parameter(classifier_name):
    
    # create an empty dictionary
    params = dict()
    if classifier_name == 'SVM':
        
        # its the degree of correct classification
        params['C'] = st.sidebar.slider('C', 0.01, 10.0)
         
    elif classifier_name == 'KNN':
        
        # no of nearest neighbours
        params['K'] = st.sidebar.slider('K', 1, 15)
        
    else:

        max_depth = st.sidebar.slider('Max_Depth', 2, 15)
        params['Max_Depth'] = max_depth # depth of each subtree in Random Forest
        n_estimators = st.sidebar.slider('N_Estimators', 1, 100)
        params['N_Estimators'] = n_estimators # no of trees
    
    return params
# Now call this function
params = get_model_parameter(get_classifier_name)


# Step 6
def get_classifier(classifier_name, params): # ui means user input
    model = None

    if classifier_name == 'SVM':
        model = SVC(C=params['C'])
    
    elif classifier_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=params['K'])
    
    else:
        model = RandomForestClassifier(n_estimators=params['N_Estimators'],
        max_depth = params['Max_Depth'], random_state= 1234)
    
    return model

model = get_classifier(get_classifier_name, params)

# Split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# check the accuracy
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {get_classifier_name}')
st.write(f'Accuracy = ', acc)

# Scatter Plot based on PCA
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c= y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)
