import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from joblib import load
import shap
import time
# Load financial statement data
#@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Train  model
def train_model(data):
    # Select features and target variable
    features = ['TA',	'R',	'IH',	'InH',	'NI',	'NI.R',	'NM',	'OM',	'ROE',	'ROA',	'EV.EBIT',	'EV.EBITDA',	'NCWC.gr',	'TA.R',	'Cash.R',	'NCWC.R',	'FA.TA',	'R_gr',	'R_exp',	'SGAE.R',	'NCWC',	'BD.TA',	'Beta',	'Div.P',	'EPS',	'EPS_gr',	'EPS_exp',	'P.E',	'PEG',	'PBV',	'RR',	'PS',	'Pos',	'Neg',	'Tone',	'Uncert',	'Litig',	'ModStrong',	'ModWeak',	'Constr']
    target = 'Class_f_nf'
    feature = ['TA',	'R',	'IH',	'InH',	'NI',	'NI.R',	'NM',	'OM',	'ROE',	'ROA',	'EV.EBIT',	'EV.EBITDA',	'NCWC.gr',	'TA.R',	'Cash.R',	'NCWC.R',	'FA.TA',	'R_gr',	'R_exp',	'SGAE.R',	'NCWC',	'BD.TA',	'Beta',	'Div.P',	'EPS',	'EPS_gr',	'EPS_exp',	'P.E',	'PEG',	'PBV',	'RR',	'PS',	'Pos',	'Neg',	'Tone',	'Uncert',	'Litig',	'ModStrong',	'ModWeak',	'Constr']
    X = data[features]
    y = data[target]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train Random Forest classifier
    if selected_train == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        #start_time = time.time()
        clf.fit(X_train, y_train)  
        #time_taken = time.time() - start_time
        #st.write(time_taken)
        joblib.dump(clf, 'FD_model_RF.joblib')
    # Train Logistic Regression classifier
    elif selected_train == 'Logistic Regression':
        #start_time = time.time()
        clf = LogisticRegression(max_iter=200).fit(X_train, y_train)
        #time_taken = time.time() - start_time
        #st.write(time_taken)
        joblib.dump(clf, 'FD_model_LR.joblib')
    # Train SVM   
    elif selected_train == 'SVM':
        clf = SVC(kernel='rbf', gamma='auto', C=1.0)
        #clf = SVC(kernel='linear', probability=True)
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'FD_model_SV.joblib')
    # Train Nayve Bayes  
    elif selected_train == 'Nayve Bayes':
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'FD_model_NB.joblib')
    

    #Test Prediction
    y_pred = clf.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    return clf, accuracy, confusion_mat, X_train, feature

# Custom styling function for DataFrame
def highlight_fraud(row):
    if row['Predicted Fraud Flag'] == 'Fraud':
        return ['background-color: red'] * len(row)
    elif row['Predicted Fraud Flag'] == 'Non Fraud':
        return ['background-color: green'] * len(row)
    else:
        return [''] * len(row)

# Streamlit Web Application
st.title('Financial Statement Fraud Detection App(PNL, Balance Sheet and Ratios)')
st.subheader('About the App')
st.write('This APP is AI trained App that is used as a first level flagging tool for determining if a financial statement i.e. PNL, \n',
             'Balance Sheet and the Ratios have been manipulated and thus requiring more attention for proffessional body to conduct further checks')

# Sidebar option to upload CSV for training
st.sidebar.header('Training Data')
selected_train= st.sidebar.selectbox('Select Training:', ['Random Forest','Logistic Regression'])
uploaded_train_file = st.sidebar.file_uploader("Upload CSV file for training", type=['csv'])

# Sidebar option to upload CSV for prediction
st.sidebar.header('Prediction Data')
selected_predic= st.sidebar.selectbox('Select Model for Prediction:', ['Random Forest', 'Logistic Regression'])
uploaded_pred_file = st.sidebar.file_uploader("Upload CSV file for prediction", type=['csv'])
st.sidebar.subheader('Visualise Predicted Data')
selected_metric = st.sidebar.selectbox('Select Metric:', ['Revenue', 'Total Assets', 'Net Income','Return on Asset','Return on Equity'])

if uploaded_train_file is not None:

    training_data = load_data(uploaded_train_file)
    
    # Call  model training after file load
    model, accuracy, confusion_matrix, X_train, features = train_model(training_data)
    
    # Display model evaluation metrics
    st.subheader('Model Evaluation Metrics')
    st.write(f'Model: {selected_train}')
    st.write(f'Accuracy: {accuracy}')
    st.write('Confusion Matrix:')
    st.write(confusion_matrix)



       # Train SHAP explainer

    
    if selected_train == 'Random Forest':
        #st.write('Hello RF',selected_train)
        #start_time = time.time()

        explainer = shap.TreeExplainer(model, X_train)
        shap_values = explainer.shap_values(X_train)
        mean_shap_values1 = np.abs(shap_values).mean(axis=0)
        mean_shap_values = np.abs(mean_shap_values1).mean(axis=1)
      
 
    elif selected_train == 'Logistic Regression':

        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_train)
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        #time_taken = time.time() - start_time
        #st.write(time_taken)
    #elif selected_train == 'SVM':
    #    st.write('Hello SV',selected_train)
    #    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    #    shap_values = explainer.shap_values(X_train)
    #    mean_shap_values = np.abs(shap_values).mean(axis=0)
    #elif selected_train == 'Nayve Bayes':
    #    st.write('Hello Nayve Bayes',selected_train)
    #    explainer = shap.KernelExplainer(model, X_train)
    #    shap_values = explainer.shap_values(X_train)
    #    mean_shap_values = np.abs(shap_values).mean(axis=0) """
    

    
    st.subheader('SHAP Explanation on Top 10 Features influencing the Training results by Magnitude of influence')
 
    sorted_indices = np.argsort(mean_shap_values)
    top_10_indices = sorted_indices[-10:]
    #st.write("Sorted indices:", top_10_indices)
    sorted_features = [features[i] for i in top_10_indices]

    fig, ax = plt.subplots()
    ax.barh(sorted_features, mean_shap_values[top_10_indices])
    #ax.barh(X_train.columns, mean_shap_values)
    ax.set_xlabel('Mean Absolute SHAP Value')
    ax.set_ylabel('Feature')
    ax.set_title('SHAP Values for Features')
    st.pyplot(fig)

    uploaded_train_file = None

if uploaded_pred_file is not None:
    prediction_data = load_data(uploaded_pred_file)
    
    # Make predictions
    if 'TA' in prediction_data.columns:
        X_pred = prediction_data[['TA',	'R',	'IH',	'InH',	'NI',	'NI.R',	'NM',	'OM',	'ROE',	'ROA',	'EV.EBIT',	'EV.EBITDA',	'NCWC.gr',	'TA.R',	'Cash.R',	'NCWC.R',	'FA.TA',	'R_gr',	'R_exp',	'SGAE.R',	'NCWC',	'BD.TA',	'Beta',	'Div.P',	'EPS',	'EPS_gr',	'EPS_exp',	'P.E',	'PEG',	'PBV',	'RR',	'PS',	'Pos',	'Neg',	'Tone',	'Uncert',	'Litig',	'ModStrong',	'ModWeak',	'Constr']]
        if selected_predic == 'Random Forest':
            model=load('FD_model_RF.joblib')
            explainer = shap.TreeExplainer(model, X_pred)
            shap_values = explainer.shap_values(X_pred)
            mean_shap_values1 = np.abs(shap_values).mean(axis=0)
            mean_shap_values = np.abs(mean_shap_values1).mean(axis=1)
        elif selected_predic == 'Logistic Regression':
            model=load('FD_model_LR.joblib')
            explainer = shap.LinearExplainer(model, X_pred)
            shap_values = explainer.shap_values(X_pred)
            mean_shap_values = np.abs(shap_values).mean(axis=0)
        elif selected_predic == 'SVM':
            model=load('FD_model_SV.joblib')
        elif selected_predic == 'Nayve Bayes':
            model=load('FD_model_NB.joblib')

        y_predic = model.predict(X_pred)
        prediction_data['Predicted Fraud Flag'] = y_predic
        st.subheader(f'Prediction Results: {selected_predic}')
        predictiion_data_final = prediction_data[['TA',	'R',	'IH',	'InH',	'NI',	'NI.R',	'NM',	'OM',	'ROE',	'ROA',	'EV.EBIT',	'EV.EBITDA',	'NCWC.gr',	'TA.R',	'Cash.R',	'NCWC.R',	'FA.TA',	'R_gr',	'R_exp',	'SGAE.R',	'NCWC',	'BD.TA',	'Beta',	'Div.P',	'EPS',	'EPS_gr',	'EPS_exp',	'P.E',	'PEG',	'PBV',	'RR',	'PS',	'Pos',	'Neg',	'Tone',	'Uncert',	'Litig',	'ModStrong',	'ModWeak',	'Constr','Predicted Fraud Flag']]
        prediction_data_styled = predictiion_data_final.style.apply(highlight_fraud, axis=1)
        st.write(prediction_data_styled)


        st.subheader('SHAP Explanation on Features influencing the Prediction results by Magnitude of influence')
        sorted_indices = np.argsort(mean_shap_values)
        top_10_indices = sorted_indices[-10:]
        sorted_features = [X_pred.columns[i] for i in top_10_indices]

        fig, ax = plt.subplots()
        ax.barh(sorted_features, mean_shap_values[top_10_indices])
        #ax.barh(X_train.columns, mean_shap_values)
        ax.set_xlabel('Mean Absolute SHAP Value')
        ax.set_ylabel('Feature')
        ax.set_title('SHAP Values for Features')
        st.pyplot(fig)

        


        # Visualizations
        st.subheader(f'Data Visualizations: {selected_metric}')

        # Generate pie chart based on selected metric
        fraud_counts = prediction_data['Predicted Fraud Flag'].value_counts()

        colors=['red','green']

        if selected_metric == 'Revenue':
            fig, ax = plt.subplots()
            ax.pie(prediction_data.groupby('Predicted Fraud Flag')['R'].sum(), labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            #st.pyplot(fig)
        elif selected_metric == 'Total Assets':
            fig, ax = plt.subplots()
            ax.pie(prediction_data.groupby('Predicted Fraud Flag')['TA'].sum(), labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        #st.pyplot(fig)
        elif selected_metric == 'Net Income':
            fig, ax = plt.subplots()
            ax.pie(prediction_data.groupby('Predicted Fraud Flag')['NI'].sum(), labels=fraud_counts.index, autopct='%1.1f%%', startangle=90,colors=colors)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        elif selected_metric == 'Return on Asset':
            fig, ax = plt.subplots()
            ax.pie(prediction_data.groupby('Predicted Fraud Flag')['ROA'].sum(), labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            #st.pyplot(fig)
        elif selected_metric == 'Return on Equity':
            fig, ax = plt.subplots()
            ax.pie(prediction_data.groupby('Predicted Fraud Flag')['ROE'].sum(), labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        #st.pyplot(fig)   
        st.pyplot(fig)

        uploaded_pred_file = None

    else:
        st.write("Prediction data should have columns 'Revenue', 'Net Income', 'Total Assets'.")

# Additional analysis and features
# Implement additional analysis or features as needed
