
Project Name: A Domain-Specific Data Science Product Development Project: Product Prototype 
Description: This Streamlit application prototype leverages machine learning models for detecting manipulated financial statements. It employs data science techniques for predictive modeling and data analysis, presenting the results in an interactive web interface.

 Requirements
- Python 3.11
- Libraries:
  - Streamlit
  - Pandas
  - Matplotlib
  - NumPy
  - scikit-learn
  - Joblib
  - SHAP
  - Others as required

 For fresh data generation
- R version 4.2.3
- RStudio(or any IDE that can run R
- Libraries:
  - dplyr  - tidyverse


 Installation
1. Create a copy of the Folder to a location of choice:  
  
2. Install required packages:  
   Ensure Python 3.11 is installed on your system. Install the necessary Python packages using pip or pip3:

   pip install streamlit pandas matplotlib numpy scikit-learn joblib shap
or
   pip3 install streamlit pandas matplotlib numpy scikit-learn joblib shap

 Configuration
No configurations apart from the above set up are required

 Usage
To run the Streamlit application, navigate to the script's directory and execute the following command:

streamlit run FraudDetectionSt_V2.py

Already saved model files can be loaded for predictions through the interface of the application

There  are 2 data files for training and predictions as labeled below

   Training data: financial_statements_data_T1_5.csv
   Predictions: financial_statements_validation_T1_5.csv 

There is a PDF file showing the full meaning of the abbreviated Feature names

 Contributing
Changes are welcome for expanding features or fixing bugs. For significant changes, please open an issue first by sending me an email at a.jonathanmensah@yahoo.com or bi53ja@student.sunderland.ac.uk to discuss your ideas. Please ensure to update me as appropriate if personal changes are made.




