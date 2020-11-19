import os
import streamlit as st 

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 
import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix,plot_roc_curve,precision_score,recall_score,precision_recall_curve,roc_auc_score,auc
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
try:
 
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union
 
    import pandas as pd
    import streamlit as st
except Exception as e:
    print(e)
import streamlit.components.v1 as stc




""" Common ML Dataset Explorer """
st.title("Machine Learning Tutorial App")
st.subheader("Explorer with Streamlit")

html_temp = """
<div style="background-color:#000080;"><p style="color:white;font-size:50px;padding:10px">ML is Awesome</p></div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.subheader("Dataset")
datasetchoice = st.radio("Do you what to use your own dataset?", ("Yes", "No"))
if datasetchoice=='No':
  def file_selector(folder_path='./datasets'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select A file",filenames)
    return os.path.join(folder_path,selected_filename)

  filename = file_selector()
  st.info("You Selected {}".format(filename))

  # Read Data
  df = pd.read_csv(filename)
  # Show Dataset

  if st.checkbox("Show Dataset"):
    st.dataframe(df)

  # Show Columns
  if st.button("Column Names"):
    st.write(df.columns)

  # Show Shape
  if st.checkbox("Shape of Dataset"):
    data_dim = st.radio("Show Dimensions By ",("Rows","Columns"))
    if data_dim == 'Rows':
      st.text("Number of Rows")
      st.write(df.shape[0])
    elif data_dim == 'Columns':
      st.text("Number of Columns")
      st.write(df.shape[1])
    else:
      st.write(df.shape)

  # Select Columns
  if st.checkbox("Select Columns To Show"):
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select",all_columns)
    new_df = df[selected_columns]
    st.dataframe(new_df)

  # Show Values
  if st.button("Value Counts"):
    st.text("Value Counts By Target/Class")
    st.write(df.iloc[:,-1].value_counts())


  # Show Datatypes
  if st.button("Data Types"):
    st.write(df.dtypes)


  # Show Summary
  if st.checkbox("Summary"):
    st.write(df.describe().T)

  ## Plot and Visualization

  st.subheader("Data Visualization")
  # Correlation
  # Seaborn Plot
  if st.checkbox("Correlation Plot[Seaborn]"):
    st.write(sns.heatmap(df.corr(),annot=True))
    st.pyplot()


  # Pie Chart
  if st.checkbox("Pie Plot"):
    all_columns_names = df.columns.tolist()
    if st.button("Generate Pie Plot"):
      st.success("Generating A Pie Plot")
      st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
      st.pyplot()

  # Count Plot
  if st.checkbox("Plot of Value Counts"):
    st.text("Value Counts By Target")
    all_columns_names = df.columns.tolist()
    primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
    selected_columns_names = st.multiselect("Select Columns",all_columns_names)
    if st.button("Plot"):
      st.text("Generate Plot")
      if selected_columns_names:
        vc_plot = df.groupby(primary_col)[selected_columns_names].count()
      else:
        vc_plot = df.iloc[:,-1].value_counts()
      st.write(vc_plot.plot(kind="bar"))
      st.pyplot()


  # Customizable Plot

  st.subheader("Customizable Plot")
  all_columns_names = df.columns.tolist()
  type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
  selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

  if st.button("Generate Plot"):
    st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

    # Plot By Streamlit
    if type_of_plot == 'area':
      cust_data = df[selected_columns_names]
      st.area_chart(cust_data)

    elif type_of_plot == 'bar':
      cust_data = df[selected_columns_names]
      st.bar_chart(cust_data)

    elif type_of_plot == 'line':
      cust_data = df[selected_columns_names]
      st.line_chart(cust_data)

    # Custom Plot 
    elif type_of_plot:
      cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
      st.write(cust_plot)
      st.pyplot()

    if st.button("End of Data Exploration"):
      st.balloons()
  st.sidebar.subheader('Choose Classifer')
  classifier_name = st.sidebar.selectbox(
      'Choose classifier',
      ('KNN', 'SVM', 'Random Forest','Logistic Regression','XGBOOST','Unsupervised Learning')
  )
  label= LabelEncoder()
  for col in df.columns:
    df[col]=label.fit_transform(df[col])



  if classifier_name == 'Unsupervised Learning':
    st.sidebar.subheader('Model Hyperparmeter')
    n_clusters= st.sidebar.number_input("number of clusters",2,10,step=1,key='clusters')
    if st.sidebar.button("classify",key='classify'):	
        sc = StandardScaler()
        X_transformed = sc.fit_transform(df)
        pca = PCA(n_components=2).fit_transform(X_transformed) # calculation Cov matrix is embeded in PCA
        kmeans = KMeans(n_clusters)
        kmeans.fit(pca)
        st.set_option('deprecation.showPyplotGlobalUse', False)
    # plt.figure(figsize=(12,10))
        plt.scatter(pca[:,0],pca[:,1], c=kmeans.labels_, cmap='rainbow')
        plt.title('CLustering Projection');
        st.pyplot()
  
  Y = df.target
  X = df.drop(columns=['target'])
  
  
  X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=8)
  
  from sklearn.preprocessing import StandardScaler
  sl=StandardScaler()
  X_trained= sl.fit_transform(X_train)
  X_tested= sl.fit_transform(X_test)
  
  class_name=['yes','no']
  
  if classifier_name == 'SVM':
      st.sidebar.subheader('Model Hyperparmeter')
      c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='c')
      kernel= st.sidebar.radio("kernel",("linear","rbf"),key='kernel')
      gamma= st.sidebar.radio("gamma(kernel coefficiency",("scale","auto"),key='gamma')
  
      metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("SVM result")
          svcclassifier= SVC(C=c,kernel=kernel,gamma=gamma)
          svcclassifier.fit(X_trained,y_train)
          y_pred= svcclassifier.predict(X_tested)
          acc= accuracy_score(y_test,y_pred)
          st.write("Accuracy:",acc.round(2))
  # 	st.write("precision_score:",precision_score(y_test,y_pred,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_pred,average='micro').round(2))
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(svcclassifier,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(svcclassifier,X_tested,y_test,normalize=False)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(svcclassifier,X_tested,y_test,normalize=False)
              st.pyplot()
          
  
  
  if classifier_name == 'Logistic Regression':
      st.sidebar.subheader('Model Hyperparmeter')
      c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='Logistic')
      max_iter= st.sidebar.slider("maximum number of iteration",100,500,key='max_item')
    
  
      metrics= st.sidebar.multiselect("Wht is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("Logistic Regression result")
          Regression= LogisticRegression(C=c,max_iter=max_iter)
          Regression.fit(X_trained,y_train)
          y_prediction= Regression.predict(X_tested)
          acc= accuracy_score(y_test,y_prediction)
          st.write("Accuracy:",acc.round(2))
          st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(Regression,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(Regression,X_tested,y_test)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(Regression,X_tested,y_test)
              st.pyplot()
          
              
  
  if classifier_name == 'Random Forest':
      st.sidebar.subheader('Model Hyperparmeter')
      n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='estimators')
      max_depth= st.sidebar.number_input("maximum depth of tree",1,20,step=1,key='max_depth')
      bootstrap= st.sidebar.radio("Boostrap sample when building trees",("True","False"),key='boostrap')
  
  
      metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("Random Forest result")
          model= RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
          model.fit(X_trained,y_train)
          y_prediction= model.predict(X_tested)
          acc= accuracy_score(y_test,y_prediction)
          st.write("Accuracy:",acc.round(2))
          st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(model,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot() 
  
  
  if classifier_name == 'KNN':
      st.sidebar.subheader('Model Hyperparmeter')
      n_neighbors= st.sidebar.number_input("Number of n_neighbors",5,30,step=1,key='neighbors')
      leaf_size= st.sidebar.slider("leaf size",30,200,key='leaf')
      weights= st.sidebar.radio("weight function used in prediction",("uniform","distance"),key='weight')
  
  
      metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("KNN result")
          model= KNeighborsClassifier(n_neighbors=n_neighbors,leaf_size=leaf_size,weights=weights)
          model.fit(X_trained,y_train)
          y_prediction= model.predict(X_tested)
          acc= accuracy_score(y_test,y_prediction)
          st.write("Accuracy:",acc.round(2))
          st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(model,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot() 
  
  
  if classifier_name == 'XGBOOST':
      st.sidebar.subheader('Model Hyperparmeter')
      n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='XGBestimators')
      seed= st.sidebar.number_input("number of the seed",1,150,step=1,key='seed')
      metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("XGBOOST result")
    
          model= xgb.XGBClassifier(n_estimators=n_estimators,seed=seed)
          model.fit(X_trained,y_train)
          y_prediction= model.predict(X_tested)
          acc= accuracy_score(y_test,y_prediction)
          st.write("Accuracy:",acc.round(2))
          st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
          st.write("ROC_AUC_score:",roc_auc_score(y_test,y_prediction,average='micro').round(2))
  
        
  
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(model,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot() 
elif datasetchoice == 'Yes': 
  data_file = st.file_uploader("Upload CSV",type=['csv'])
# if st.button("Process"):
#   if data_file is not None:
#     file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
#     st.write(file_details)
#     df = pd.read_csv(data_file)
#     st.dataframe(df)
  st.write("Note:if you want to do classification make sure you have target attributes")    	
  def file_selector(dataset):
    if dataset is not None:
      file_details = {"Filename":dataset.name,"FileType":dataset.type,"FileSize":dataset.size}
      st.write(file_details)
      df = pd.read_csv(dataset)
      return df	
  df = file_selector(data_file)	
  st.dataframe(df)
    
    
  # def file_selector(folder_path='./datasets'):
  # 	filenames = os.listdir(folder_path)
  # 	selected_filename = st.selectbox("Select A file",filenames)
  # 	return os.path.join(folder_path,selected_filename)

  # filename = file_selector()
  # st.info("You Selected {}".format(filename))

  # # Read Data
  # df = pd.read_csv(filename)
  # # Show Dataset

  if st.checkbox("Show Dataset"):
    st.dataframe(df)

  # Show Columns
  if st.button("Column Names"):
    st.write(df.columns)

  # Show Shape
  if st.checkbox("Shape of Dataset"):
    data_dim = st.radio("Show Dimensions By ",("Rows","Columns"))
    if data_dim == 'Rows':
      st.text("Number of Rows")
      st.write(df.shape[0])
    elif data_dim == 'Columns':
      st.text("Number of Columns")
      st.write(df.shape[1])
    else:
      st.write(df.shape)

  # Select Columns
  if st.checkbox("Select Columns To Show"):
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select",all_columns)
    new_df = df[selected_columns]
    st.dataframe(new_df)

  # Show Values
  if st.button("Value Counts"):
    st.text("Value Counts By Target/Class")
    st.write(df.iloc[:,-1].value_counts())


  # Show Datatypes
  if st.button("Data Types"):
    st.write(df.dtypes)


  # Show Summary
  if st.checkbox("Summary"):
    st.write(df.describe().T)

  ## Plot and Visualization

  st.subheader("Data Visualization")
  # Correlation
  # Seaborn Plot
  if st.checkbox("Correlation Plot[Seaborn]"):
    st.write(sns.heatmap(df.corr(),annot=True))
    st.pyplot()


  # Pie Chart
  if st.checkbox("Pie Plot"):
    all_columns_names = df.columns.tolist()
    if st.button("Generate Pie Plot"):
      st.success("Generating A Pie Plot")
      st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
      st.pyplot()

  # Count Plot
  if st.checkbox("Plot of Value Counts"):
    st.text("Value Counts By Target")
    all_columns_names = df.columns.tolist()
    primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
    selected_columns_names = st.multiselect("Select Columns",all_columns_names)
    if st.button("Plot"):
      st.text("Generate Plot")
      if selected_columns_names:
        vc_plot = df.groupby(primary_col)[selected_columns_names].count()
      else:
        vc_plot = df.iloc[:,-1].value_counts()
      st.write(vc_plot.plot(kind="bar"))
      st.pyplot()


  # Customizable Plot

  st.subheader("Customizable Plot")
  all_columns_names = df.columns.tolist()
  type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
  selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

  if st.button("Generate Plot"):
    st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

    # Plot By Streamlit
    if type_of_plot == 'area':
      cust_data = df[selected_columns_names]
      st.area_chart(cust_data)

    elif type_of_plot == 'bar':
      cust_data = df[selected_columns_names]
      st.bar_chart(cust_data)

    elif type_of_plot == 'line':
      cust_data = df[selected_columns_names]
      st.line_chart(cust_data)

    # Custom Plot 
    elif type_of_plot:
      cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
      st.write(cust_plot)
      st.pyplot()

    if st.button("End of Data Exploration"):
      st.balloons()
  st.sidebar.subheader('Choose Classifer')
  classifier_name = st.sidebar.selectbox(
      'Choose classifier',
      ('KNN', 'SVM', 'Random Forest','Logistic Regression','XGBOOST','Unsupervised Learning')
  )
  label= LabelEncoder()
  for col in df.columns:
    df[col]=label.fit_transform(df[col])



  if classifier_name == 'Unsupervised Learning':
    st.sidebar.subheader('Model Hyperparmeter')
    n_clusters= st.sidebar.number_input("number of clusters",2,10,step=1,key='clusters')
    if st.sidebar.button("classify",key='classify'):	
        sc = StandardScaler()
        X_transformed = sc.fit_transform(df)
        pca = PCA(n_components=2).fit_transform(X_transformed) # calculation Cov matrix is embeded in PCA
        kmeans = KMeans(n_clusters)
        kmeans.fit(pca)
        st.set_option('deprecation.showPyplotGlobalUse', False)
    # plt.figure(figsize=(12,10))
        plt.scatter(pca[:,0],pca[:,1], c=kmeans.labels_, cmap='rainbow')
        plt.title('CLustering Projection');
        st.pyplot()
  
  Y = df.target
  X = df.drop(columns=['target'])
  
  
  X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=8)
  
  from sklearn.preprocessing import StandardScaler
  sl=StandardScaler()
  X_trained= sl.fit_transform(X_train)
  X_tested= sl.fit_transform(X_test)
  
  class_name=['yes','no']
  
  if classifier_name == 'SVM':
      st.sidebar.subheader('Model Hyperparmeter')
      c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='c')
      kernel= st.sidebar.radio("kernel",("linear","rbf"),key='kernel')
      gamma= st.sidebar.radio("gamma(kernel coefficiency",("scale","auto"),key='gamma')
  
      metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("SVM result")
          svcclassifier= SVC(C=c,kernel=kernel,gamma=gamma)
          svcclassifier.fit(X_trained,y_train)
          y_pred= svcclassifier.predict(X_tested)
          acc= accuracy_score(y_test,y_pred)
          st.write("Accuracy:",acc.round(2))
  # 	st.write("precision_score:",precision_score(y_test,y_pred,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_pred,average='micro').round(2))
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(svcclassifier,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(svcclassifier,X_tested,y_test,normalize=False)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(svcclassifier,X_tested,y_test,normalize=False)
              st.pyplot()
          
  
  
  if classifier_name == 'Logistic Regression':
      st.sidebar.subheader('Model Hyperparmeter')
      c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='Logistic')
      max_iter= st.sidebar.slider("maximum number of iteration",100,500,key='max_item')
    
  
      metrics= st.sidebar.multiselect("Wht is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("Logistic Regression result")
          Regression= LogisticRegression(C=c,max_iter=max_iter)
          Regression.fit(X_trained,y_train)
          y_prediction= Regression.predict(X_tested)
          acc= accuracy_score(y_test,y_prediction)
          st.write("Accuracy:",acc.round(2))
          st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(Regression,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(Regression,X_tested,y_test)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(Regression,X_tested,y_test)
              st.pyplot()
          
              
  
  if classifier_name == 'Random Forest':
      st.sidebar.subheader('Model Hyperparmeter')
      n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='estimators')
      max_depth= st.sidebar.number_input("maximum depth of tree",1,20,step=1,key='max_depth')
      bootstrap= st.sidebar.radio("Boostrap sample when building trees",("True","False"),key='boostrap')
  
  
      metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("Random Forest result")
          model= RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
          model.fit(X_trained,y_train)
          y_prediction= model.predict(X_tested)
          acc= accuracy_score(y_test,y_prediction)
          st.write("Accuracy:",acc.round(2))
          st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(model,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot() 
  
  
  if classifier_name == 'KNN':
      st.sidebar.subheader('Model Hyperparmeter')
      n_neighbors= st.sidebar.number_input("Number of n_neighbors",5,30,step=1,key='neighbors')
      leaf_size= st.sidebar.slider("leaf size",30,200,key='leaf')
      weights= st.sidebar.radio("weight function used in prediction",("uniform","distance"),key='weight')
  
  
      metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("KNN result")
          model= KNeighborsClassifier(n_neighbors=n_neighbors,leaf_size=leaf_size,weights=weights)
          model.fit(X_trained,y_train)
          y_prediction= model.predict(X_tested)
          acc= accuracy_score(y_test,y_prediction)
          st.write("Accuracy:",acc.round(2))
          st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(model,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot() 
  
  
  if classifier_name == 'XGBOOST':
      st.sidebar.subheader('Model Hyperparmeter')
      n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='XGBestimators')
      seed= st.sidebar.number_input("number of the seed",1,150,step=1,key='seed')
      metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
  
      if st.sidebar.button("classify",key='classify'):
          st.subheader("XGBOOST result")
    
          model= xgb.XGBClassifier(n_estimators=n_estimators,seed=seed)
          model.fit(X_trained,y_train)
          y_prediction= model.predict(X_tested)
          acc= accuracy_score(y_test,y_prediction)
          st.write("Accuracy:",acc.round(2))
          st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
          st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
          st.write("ROC_AUC_score:",roc_auc_score(y_test,y_prediction,average='micro').round(2))
  
        
  
          if 'confusion matrix' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('confusion matrix')
              plot_confusion_matrix(model,X_tested,y_test)
              st.pyplot()
          if 'roc_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('plot_roc_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot()
          if 'precision_recall_curve' in metrics:
              st.set_option('deprecation.showPyplotGlobalUse', False)
              st.subheader('precision_recall_curve')
              plot_roc_curve(model,X_tested,y_test)
              st.pyplot() 
                    
        















































# import os
# import streamlit as st 
 
# # EDA Pkgs
# import pandas as pd 
 
# # Viz Pkgs
# import matplotlib.pyplot as plt 
# import matplotlib
# import io
# matplotlib.use('Agg')
# import seaborn as sns 
 
# def main():
#     """ Common ML Dataset Explorer """
#     st.title("Machine Learning Tutorial")
#     st.subheader("Datasets For ML Explorer with Streamlit")
# #     file = st.file_uploader("please Upload your dataset",type=['.csv'])
#     st.set_option('deprecation.showfileUploaderEncoding', False)
#     html_temp = """
#     <div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Streamlit is Awesome</p></div>
#     """
#     import csv 
 
    
#     st.markdown(html_temp,unsafe_allow_html=True)
#     file_buffer = st.file_uploader("Choose a CSV Log File...", type="csv", encoding = None)
#     dataset = pd.read_csv(file_buffer)
#     with open(file_buffer,'r') as csv_file: #Opens the file in read mode
#         csv_reader = csv.reader(csv_file)
#     if dataset is not None:
#         df = open(dataset)  
#         st.write(df)
           
 
#     # Show Columns
#     if st.checkbox("Show Dataset"):
#         number = st.number_input("Number of Rows to View")
#         st.dataframe(df.head(number))
#     if st.button("Column Names"):
#         st.write(df.columns)
 
#     # Show Shape
#     if st.checkbox("Shape of Dataset"):
#         data_dim = st.radio("Show Dimension By ",("Rows","Columns"))
#         if data_dim == 'Rows':
#             st.text("Number of Rows")
#             st.write(df.shape[0])
#         elif data_dim == 'Columns':
#             st.text("Number of Columns")
#             st.write(df.shape[1])
#         else:
#             st.write(df.shape)
 
#     # Select Columns
#     if st.checkbox("Select Columns To Show"):
#         all_columns = df.columns.tolist()
#         selected_columns = st.multiselect("Select",all_columns)
#         new_df = df[selected_columns]
#         st.dataframe(new_df)
    
#     # Show Values
#     if st.button("Value Counts"):
#         st.text("Value Counts By Target/Class")
#         st.write(df.iloc[:,-1].value_counts())
 
 
#     # Show Datatypes
#     if st.button("Data Types"):
#         st.write(df.dtypes)
 
 
#     # Show Summary
#     if st.checkbox("Summary"):
#         st.write(df.describe().T)
 
#     ## Plot and Visualization
 
#     st.subheader("Data Visualization")
#     # Correlation
#     # Seaborn Plot
#     if st.checkbox("Correlation Plot[Seaborn]"):
#         st.write(sns.heatmap(df.corr(),annot=True))
#         st.pyplot()
 
    
#     # Pie Chart
#     if st.checkbox("Pie Plot"):
#         all_columns_names = df.columns.tolist()
#         if st.button("Generate Pie Plot"):
#             st.success("Generating A Pie Plot")
#             st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
#             st.pyplot()
 
#     # Count Plot
#     if st.checkbox("Plot of Value Counts"):
#         st.text("Value Counts By Target")
#         all_columns_names = df.columns.tolist()
#         primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
#         selected_columns_names = st.multiselect("Select Columns",all_columns_names)
#         if st.button("Plot"):
#             st.text("Generate Plot")
#             if selected_columns_names:
#                 vc_plot = df.groupby(primary_col)[selected_columns_names].count()
#             else:
#                 vc_plot = df.iloc[:,-1].value_counts()
#             st.write(vc_plot.plot(kind="bar"))
#             st.pyplot()
 
 
#     # Customizable Plot
 
#     st.subheader("Customizable Plot")
#     all_columns_names = df.columns.tolist()
#     type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
#     selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
 
#     if st.button("Generate Plot"):
#         st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
 
#         # Plot By Streamlit
#         if type_of_plot == 'area':
#             cust_data = df[selected_columns_names]
#             st.area_chart(cust_data)
 
#         elif type_of_plot == 'bar':
#             cust_data = df[selected_columns_names]
#             st.bar_chart(cust_data)
 
#         elif type_of_plot == 'line':
#             cust_data = df[selected_columns_names]
#             st.line_chart(cust_data)
 
#         # Custom Plot 
#         elif type_of_plot:
#             cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
#             st.write(cust_plot)
#             st.pyplot()
 
#     if st.button("Thanks"):
#         st.balloons()
 
#     st.sidebar.header("About App")
#     st.sidebar.info("A Simple EDA App for Exploring Common ML Dataset")
 
#     st.sidebar.header("Get Datasets")
#     st.sidebar.markdown("[Common ML Dataset Repo]("")")
#     #
#     # st.sidebar.header("About")
#     # st.sidebar.info("Jesus Saves@JCharisTech")
#     # st.sidebar.text("Built with Streamlit")
#     # st.sidebar.text("Maintained by Jesse JCharis")
 
 
# if __name__ == '__main__':
#     main()



































# import pickle
# #pickle.dump(kmeans,open('unsupervisedmodels.pkl','wb'))
# import streamlit as st
# import pickle
# # import numpy as np
# # from sklearn.cluster import KMeans
# # #kmeans=pickle.load(open('unsupervisedmodels.pkl','rb')) 
# # from sklearn import datasets
# # from sklearn.manifold import TSNE
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import MinMaxScaler
# # scaler = MinMaxScaler()
# # transformed = scaler.fit_transform(x)
# # # Plotting 2d t-Sne
# # x_axis = transformed[:,0]
# # y_axis = transformed[:,1]

# # kmeans = KMeans(n_clusters=4, random_state=42,n_jobs=-1)
# # #y_pred =kmeans.fit_predict(transformed)

# # # def predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing):
# # #     input=np.array([[CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing]]).astype(np.float64)
# # #     prediction=kmeans.predict(input)
# # #     return prediction

# # st.title("Records of countries classified in the clusters")
# # html_temp = """
# # <div style="background-color:#025246 ;padding:12px">
# # <h2 style="color:white;text-align:center;">Unsupervised App </h2>
# # </div>
# # """
# # st.markdown(html_temp, unsafe_allow_html=True)
# # CountryName = st.text_input("CountryName","Type Here",key='0')
# # StringencyLegacyIndexForDisplay = st.text_input("StringencyLegacyIndexForDisplay","Type Here",key='1')
# # StringencyIndexForDisplay = st.text_input("StringencyIndexForDisplay","Type Here",key='2')
# # StringencyIndex = st.text_input("StringencyIndex","Type Here",key='3')
# # StringencyLegacyIndex = st.text_input("StringencyLegacyIndex","Type Here",key='4')
# # ContainmentHealthIndexForDisplay = st.text_input("ContainmentHealthIndexForDisplay","Type Here",key='5')
# # GovernmentResponseIndexForDisplay = st.text_input("GovernmentResponseIndexForDisplay","Type Here",key='6')
# # ContainmentHealthIndex = st.text_input("ContainmentHealthIndex","Type Here",key='7')
# # ConfirmedCases = st.text_input("ConfirmedCases","Type Here",key='8')
# # ConfirmedDeaths = st.text_input("ConfirmedDeaths","Type Here",key='9')
# # EconomicSupportIndexForDisplay = st.text_input("EconomicSupportIndexForDisplay","Type Here",key='9')
# # E2_Debtcontractrelief = st.text_input("E2_Debtcontractrelief","Type Here",key='10')
# # EconomicSupportIndex = st.text_input("EconomicSupportIndex","Type Here",key='11')
# # C3_Cancelpublicevents = st.text_input("C3_Cancelpublicevents","Type Here",key='12')
# # C1_Schoolclosing = st.text_input("C1_Schoolclosing","Type Here",key='13')

# # if st.button("Predict"):
# #   output=predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing)
# #   st.success('This country located in this cluster {}'.format(output))

# # -*- coding: utf-8 -*-
# """Assignment3.ipynb
# """

# import pandas as pd

# data= pd.read_csv('https://raw.githubusercontent.com/Diane10/ML/master/assignment3.csv')
# # data.info()

# # data.isnull().sum()

# null_counts = data.isnull().sum().sort_values()
# selected = null_counts[null_counts < 8000 ]

# percentage = 100 * data.isnull().sum() / len(data)


# data_types = data.dtypes
# # data_types

# missing_values_table = pd.concat([null_counts, percentage, data_types], axis=1)
# # missing_values_table

# col=['CountryName','Date','StringencyLegacyIndexForDisplay','StringencyIndexForDisplay','ContainmentHealthIndexForDisplay','GovernmentResponseIndexForDisplay',
# 'EconomicSupportIndexForDisplay','C8_International travel controls','C1_School closing','C3_Cancel public events','C2_Workplace closing','C4_Restrictions on gatherings',
# 'C6_Stay at home requirements','C7_Restrictions on internal movement','H1_Public information campaigns','E1_Income support','C5_Close public transport','E2_Debt/contract relief','StringencyLegacyIndex','H3_Contact tracing','StringencyIndex','ContainmentHealthIndex','E4_International support','EconomicSupportIndex','E3_Fiscal measures','H5_Investment in vaccines','ConfirmedCases','ConfirmedDeaths']

# newdataset=data[col]
# newdataset= newdataset.dropna()

# from sklearn.preprocessing import LabelEncoder
# newdataset['CountryName']=LabelEncoder().fit_transform(newdataset['CountryName'])


# # # map features to their absolute correlation values
# # corr = newdataset.corr().abs()

# # # set equality (self correlation) as zero
# # corr[corr == 1] = 0

# # # of each feature, find the max correlation
# # # and sort the resulting array in ascending order
# # corr_cols = corr.max().sort_values(ascending=False)

# # # display the highly correlated features
# # display(corr_cols[corr_cols > 0.9])

# # len(newdataset)

# X=newdataset[['CountryName','StringencyLegacyIndexForDisplay','StringencyIndexForDisplay',	'StringencyIndex','StringencyLegacyIndex','ContainmentHealthIndexForDisplay','ContainmentHealthIndex','GovernmentResponseIndexForDisplay','ConfirmedCases','ConfirmedDeaths','EconomicSupportIndexForDisplay','E2_Debt/contract relief','EconomicSupportIndex','C3_Cancel public events','C1_School closing']]
# # X=newdataset[['CountryName','StringencyLegacyIndexForDisplay','StringencyIndexForDisplay',	'StringencyIndex','StringencyLegacyIndex','ContainmentHealthIndexForDisplay','ContainmentHealthIndex','GovernmentResponseIndexForDisplay','ConfirmedCases','ConfirmedDeaths']]

# # df_first_half = X[:1000]
# # df_second_half = X[1000:]

# # """Feature selector that removes all low-variance features."""

# from sklearn.feature_selection import VarianceThreshold

# selector = VarianceThreshold()
# x= selector.fit_transform(X)

# df_first_half = x[:5000]
# df_second_half = x[5000:]

# # """Create clusters/classes of similar records using features selected in (1),  use an unsupervised learning algorithm of your choice."""

# # Commented out IPython magic to ensure Python compatibility.
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from matplotlib import pyplot as plt
# import streamlit as st

# # wcss=[]
# # for i in range(1,11):
# #     kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
# #     kmeans.fit(x)
# #     wcss.append(kmeans.inertia_)
# # st.set_option('deprecation.showPyplotGlobalUse', False)
# # plt.plot(range(1,11),wcss)
# # plt.title('The Elbow Method')
# # plt.xlabel('Number of Clusters')
# # plt.ylabel('WCSS')
# # plt.show()
# # st.pyplot()

# model = KMeans(n_clusters = 6)

# pca = PCA(n_components=2).fit(x)
# pca_2d = pca.transform(x)

# model.fit(pca_2d)

# labels = model.predict(pca_2d)
# # labels
# # predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])

# # pca = PCA(n_components=2).fit(df_first_half)
# # pca_2d = pca.transform(df_first_half)
# # pca_2d

# xs = pca_2d[:, 0]
# ys = pca_2d[:, 1]
# plt.scatter(xs, ys, c = labels)
# plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

# kmeans = KMeans(n_clusters=10)
# kmeans.fit(df_first_half)
# plt.scatter(df_first_half[:,0],df_first_half[:,1], c=kmeans.labels_, cmap='rainbow')

# range_n_clusters = [2, 3, 4, 5, 6]

# # from sklearn.metrics import silhouette_samples, silhouette_score
# # import matplotlib.cm as cm
# # import numpy as np

# # for n_clusters in range_n_clusters:
# #     # Create a subplot with 1 row and 2 columns
# #     fig, (ax1, ax2) = plt.subplots(1, 2)
# #     fig.set_size_inches(18, 7)

# #     # The 1st subplot is the silhouette plot
# #     # The silhouette coefficient can range from -1, 1 but in this example all
# #     # lie within [-0.1, 1]
# #     ax1.set_xlim([-0.1, 1])
# #     # The (n_clusters+1)*10 is for inserting blank space between silhouette
# #     # plots of individual clusters, to demarcate them clearly.
# #     ax1.set_ylim([0, len(pca_2d) + (n_clusters + 1) * 10])

# #     # Initialize the clusterer with n_clusters value and a random generator
# #     # seed of 10 for reproducibility.
# #     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
# #     cluster_labels = clusterer.fit_predict(pca_2d)

# #     # The silhouette_score gives the average value for all the samples.
# #     # This gives a perspective into the density and separation of the formed
# #     # clusters
# #     silhouette_avg = silhouette_score(pca_2d, cluster_labels)
# #     print("For n_clusters =", n_clusters,
# #           "The average silhouette_score is :", silhouette_avg)

# #     # Compute the silhouette scores for each sample
# #     sample_silhouette_values = silhouette_samples(pca_2d, cluster_labels)
# #     y_lower = 10
# #     for i in range(n_clusters):
# #         # Aggregate the silhouette scores for samples belonging to
# #         # cluster i, and sort them
# #         ith_cluster_silhouette_values = \
# #             sample_silhouette_values[cluster_labels == i]

# #         ith_cluster_silhouette_values.sort()

# #         size_cluster_i = ith_cluster_silhouette_values.shape[0]
# #         y_upper = y_lower + size_cluster_i

# #         color = cm.nipy_spectral(float(i) / n_clusters)
# #         ax1.fill_betweenx(np.arange(y_lower, y_upper),
# #                           0, ith_cluster_silhouette_values,
# #                           facecolor=color, edgecolor=color, alpha=0.7)

# #         # Label the silhouette plots with their cluster numbers at the middle
# #         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

# #         # Compute the new y_lower for next plot
# #         y_lower = y_upper + 10  # 10 for the 0 samples

# #     ax1.set_title("The silhouette plot for the various clusters.")
# #     ax1.set_xlabel("The silhouette coefficient values")
# #     ax1.set_ylabel("Cluster label")

# #     # The vertical line for average silhouette score of all the values
# #     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

# #     ax1.set_yticks([])  # Clear the yaxis labels / ticks
# #     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# #     # 2nd Plot showing the actual clusters formed
# #     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
# #     ax2.scatter( pca_2d[:, 0], pca_2d[:, 1], marker='.', s=30, lw=0, alpha=0.7,
# #                 c=colors, edgecolor='k')
# #     # Labeling the clusters
# #     centers = clusterer.cluster_centers_
# #     # Draw white circles at cluster centers
# #     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
# #                 c="white", alpha=1, s=200, edgecolor='k')

# #     for i, c in enumerate(centers):
# #         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
# #                     s=50, edgecolor='k')

# #     ax2.set_title("The visualization of the clustered data.")
# #     ax2.set_xlabel("Feature space for the 1st feature")
# #     ax2.set_ylabel("Feature space for the 2nd feature")

# #     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
# #                   "with n_clusters = %d" % n_clusters),
# #                  fontsize=14, fontweight='bold')
# # plt.show()

# #km.cluster_centers_

# from sklearn import datasets
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# transformed = scaler.fit_transform(x)
# # Plotting 2d t-Sne
# x_axis = transformed[:,0]
# y_axis = transformed[:,1]

# kmeans = KMeans(n_clusters=4, random_state=42,n_jobs=-1)
# y_pred =kmeans.fit_predict(transformed)

# predicted_label = kmeans.predict([[7,7.2, 3.5, 0.8, 1.6,7.2, 3.5, 0.8, 1.6,7.2, 3.5, 0.8, 1.67, 7.2, 3.5]])
# predicted_label

# # from sklearn.manifold import TSNE
# # tsne = TSNE(random_state=17)

# # X_tsne = tsne.fit_transform(transformed)

# # plt.figure(figsize=(12,10))
# # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, 
# #             edgecolor='none', alpha=0.7, s=40,
# #             cmap=plt.cm.get_cmap('nipy_spectral', 10))
# # plt.colorbar()
# # plt.title('cluster. t-SNE projection');

# # pca = PCA(n_components=2)
# # X_reduced = pca.fit_transform(transformed)

# # print('Projecting %d-dimensional data to 2D' % X.shape[1])

# # plt.figure(figsize=(12,10))
# # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, 
# #             edgecolor='none', alpha=0.7, s=40,
# #             cmap=plt.cm.get_cmap('nipy_spectral', 10))
# # plt.colorbar()
# # plt.title('cluster. PCA projection');
# # st.pyplot()

# # """https://www.kaggle.com/kashnitsky/topic-7-unsupervised-learning-pca-and-clustering"""

# # import seaborn as sns

# # import pickle
# # pickle.dump(kmeans,open('unsupervisedmodels.pkl','wb'))

# # """Create a platform where new records of countries can be classified in the clusters"""



# # Commented out IPython magic to ensure Python compatibility.
# # %%writefile app.py
# import streamlit as st
# import pickle
# import numpy as np

# # kmeans=pickle.load(open('unsupervisedmodels.pkl','rb')) 


# def predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing):
#     input=np.array([[CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing]]).astype(np.float64)
#     prediction=kmeans.predict(input)
#     return prediction

# def main():
#     st.title("Records of countries classified in the clusters")
#     html_temp = """
#     <div style="background-color:#025246 ;padding:10px">
#     <h2 style="color:white;text-align:center;">Unsupervised ML App </h2>
#     </div>
#     """
#     st.markdown(html_temp, unsafe_allow_html=True)
#     CountryName = st.text_input("CountryName","Type Here",key='0')
#     StringencyLegacyIndexForDisplay = st.text_input("StringencyLegacyIndexForDisplay","Type Here",key='1')
#     StringencyIndexForDisplay = st.text_input("StringencyIndexForDisplay","Type Here",key='2')
#     StringencyIndex = st.text_input("StringencyIndex","Type Here",key='3')
#     StringencyLegacyIndex = st.text_input("StringencyLegacyIndex","Type Here",key='4')
#     ContainmentHealthIndexForDisplay = st.text_input("ContainmentHealthIndexForDisplay","Type Here",key='5')
#     GovernmentResponseIndexForDisplay = st.text_input("GovernmentResponseIndexForDisplay","Type Here",key='6')
#     ContainmentHealthIndex = st.text_input("ContainmentHealthIndex","Type Here",key='7')
#     ConfirmedCases = st.text_input("ConfirmedCases","Type Here",key='8')
#     ConfirmedDeaths = st.text_input("ConfirmedDeaths","Type Here",key='9')
#     EconomicSupportIndexForDisplay = st.text_input("EconomicSupportIndexForDisplay","Type Here",key='9')
#     E2_Debtcontractrelief = st.text_input("E2_Debtcontractrelief","Type Here",key='10')
#     EconomicSupportIndex = st.text_input("EconomicSupportIndex","Type Here",key='11')
#     C3_Cancelpublicevents = st.text_input("C3_Cancelpublicevents","Type Here",key='12')
#     C1_Schoolclosing = st.text_input("C1_Schoolclosing","Type Here",key='13')

#     safe_html="""  
#       <div style="background-color:#F4D03F;padding:10px >
#        <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
#        </div>
#     """
#     danger_html="""  
#       <div style="background-color:#F08080;padding:10px >
#        <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
#        </div>
#     """

#     if st.button("Predict"):
#         output=predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing)
#         st.success('This country located in this cluster {}'.format(output))


# if __name__=='__main__':
#     main()



