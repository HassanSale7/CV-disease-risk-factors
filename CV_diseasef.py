#!/usr/bin/env python
# coding: utf-8

# In[67]:


# for basic operations
import numpy as np
import pandas as pd
import pandas_profiling
from IPython import get_ipython
ipython = get_ipython()
# for data visualizations
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# for advanced visualizations 
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from bubbly.bubbly import bubbleplot
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
pd.options.mode.chained_assignment = None


# In[68]:


st.title("CardioVascular diseases and its Risk Factors")


# In[69]:


from PIL import Image


# In[70]:


image = Image.open(r'C:\Users\hassan\Desktop\health care analytics project\heart_attack.gif')
st.image(image, caption='heart attack feeling')


# In[71]:


# reading the data
data = pd.read_csv('heart.csv')
# getting the shape
data.shape


# In[72]:



# let's change the names of the  columns for better understanding

data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

data.columns


# In[73]:


data.head()


# In[74]:


def clean_data(data):
    
    data['sex'][data['sex'] == 0] = 'female'
    data['sex'][data['sex'] == 1] = 'male'

    data['chest_pain_type'][data['chest_pain_type'] == 1] = 'typical angina'
    data['chest_pain_type'][data['chest_pain_type'] == 2] = 'atypical angina'
    data['chest_pain_type'][data['chest_pain_type'] == 3] = 'non-anginal pain'
    data['chest_pain_type'][data['chest_pain_type'] == 4] = 'asymptomatic'

    data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
    data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

    data['rest_ecg'][data['rest_ecg'] == 0] = 'normal'
    data['rest_ecg'][data['rest_ecg'] == 1] = 'ST-T wave abnormality'
    data['rest_ecg'][data['rest_ecg'] == 2] = 'left ventricular hypertrophy'

    data['exercise_induced_angina'][data['exercise_induced_angina'] == 0] = 'no'
    data['exercise_induced_angina'][data['exercise_induced_angina'] == 1] = 'yes'

    data['st_slope'][data['st_slope'] == 1] = 'upsloping'
    data['st_slope'][data['st_slope'] == 2] = 'flat'
    data['st_slope'][data['st_slope'] == 3] = 'downsloping'

    data['thalassemia'][data['thalassemia'] == 1] = 'normal'
    data['thalassemia'][data['thalassemia'] == 2] = 'fixed defect'
    data['thalassemia'][data['thalassemia'] == 3] = 'reversable defect'
    
    return data


# In[75]:


def change_dtypes(data):
    """
    Function will be used to convert features to appropriate type.

    Parameters
    ----------
    data: DataFrame
        A dataframe containing the heart disease data.

    Returns
    -------
    data: DataFrame
        A dataframe containing the cleansed heart disease data.
    """
    data['sex'] = data['sex'].astype('object')
    data['chest_pain_type'] = data['chest_pain_type'].astype('object')
    data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype('object')
    data['rest_ecg'] = data['rest_ecg'].astype('object')
    data['exercise_induced_angina'] = data['exercise_induced_angina'].astype('object')
    data['st_slope'] = data['st_slope'].astype('object')
    data['thalassemia'] = data['thalassemia'].astype('object')
    
    return data


# In[76]:


data = clean_data(data)
data = change_dtypes(data)


# In[77]:


def plot_single_feature(data, feataure):
    """
    This function will be used to plot a single feature.

    Every feature's type will be first evaluated and then the
    feature's distribution will be graphed accordingly.

    Rules for single variable visualizations:
    * Numerical variables will be represented by histograms.
    * The visualizations for numerical variables will have "Frequency" as the y-axis label.
    * Categorical variables will be represented by bar charts.
    * The visualizations for categorical variables will have "Count" as the y-axis label. 

    Parameters
    ----------
    data: DataFrame
        A dataframe containing the heart disease data.
        
    feature: str
        The feature whose data needs to be plotted.

    Returns
    -------
    None
    """
    fig = None
    xaxis_type=None
    yaxis_title=""

    # Switching int features with low cardinality to object:
    data["num_major_vessels"] = data["num_major_vessels"].astype("object")
    data["target"] = data["target"].astype("object")
    # Check feature type and plot appropriately:
    if data[feature].dtype == 'int64' or data[feature].dtype == 'float64':
        #TODO(Sayar) Add slider widget here:
        fig = px.histogram(x=data[feature].values, nbins=0)

        yaxis_title = "Frequency"

    elif data[feature].dtype == 'object':
        fig = px.bar(y=data[feature].value_counts(), 
                     x=data[feature].value_counts().index.astype(str), 
                     color=data[feature].value_counts().index.astype(str), 
                     text=data[feature].value_counts())

        xaxis_type = "category"
        yaxis_title = "Count"

    fig.update_xaxes(title=feature)
    fig.update_yaxes(title=yaxis_title)
    fig.update_layout(showlegend=False, 
                      title="Distribution of {}".format(feature), 
                      xaxis_type=xaxis_type)
    st.plotly_chart(fig)
    return


# In[78]:


# Iterating over every feature and plotting it:
for feature in data.columns:
    plot_single_feature(data, feature)


# In[79]:


def plot_numerical_numerical(data, feature_1, feature_2):
    """Plots numerical vs numerical features"""
    fig = px.scatter(data, feature_1, feature_2)
    fig.update_layout(title="Plot of {} vs. {}".format(feature_1, 
                                                       feature_2))
    st.plotly_chart(fig)

def plot_numerical_categorical(data, feature_1, feature_2):
    """Plots numerical vs categorical features"""
    x_var, y_var = feature_1, feature_2
    # feature_1 is passed into x_var. If it is not categorical, 
    # we switch it with y_var:
    if data[feature_1].dtypes == "int64" or data[feature_1].dtypes == "float64":
        x_var,y_var = y_var,x_var

    fig = px.box(data, 
                 x=x_var, 
                 y=y_var, 
                 color=x_var)
                 
    fig.update_layout(title="Plot of {} vs. {}".format(x_var, y_var))

    st.plotly_chart(fig)


# In[81]:


def plot_categorical_categorical(data, feature_1, feature_2):
    """Plots categorical vs categorical features"""
    fig = px.parallel_categories(data, 
                                 dimensions=[feature_1, feature_2], 
                                 )
    fig.update_layout(title="Plot of {} vs. {}".format(feature_1, feature_2))
    st.plotly_chart(fig)

def plot_dual_features(data, feature_1, feature_2):
    """
    This function will be used to plot feature interactions between
    two features.

    Rules for feature interaction visualization:

    * Only two variables can be used for this visualization.
    * Both variables have to be different.
    * For numerical vs numerical visuals, we will be using scatter plots.
    * For numerical vs categorical visuals, we will be using box plots.
    * For categorical vs categorical visuals, we will be using scatter plots.

    Parameters
    ----------
    data: DataFrame
        A dataframe containing the heart disease data.
    
    feature_1: str
        The first feature to be used in the plot.

    feature_2: str
        The second feature to be used in the plot.

    Returns
    -------
    None
    """
    # Cannot allow same feature plots:
    if feature_1 == feature_2:
        raise ValueError("Please select two different features.")

    # Changed to object type because of low cardinality:
    data["num_major_vessels"] = data["num_major_vessels"].astype("object")
    data["target"] = data["target"].astype("object")
    feature_1_type = str(data[feature_1].dtype)
    feature_2_type = str(data[feature_2].dtype)

    # Dictionary to hash the appropriate function object:
    switch_dict = {
        ("int64", "float64"): plot_numerical_numerical, 
        ("float64", "int64"): plot_numerical_numerical,
        ("float64", "float64"): plot_numerical_numerical,
        ("int64", "int64"): plot_numerical_numerical,
        ("int64", "object"): plot_numerical_categorical,
        ("float64", "object"): plot_numerical_categorical,
        ("object", "int64"): plot_numerical_categorical,
        ("object", "float64"): plot_numerical_categorical,
        ("object", "object"): plot_categorical_categorical
    }

    # Calling function object:
    switch_dict[(feature_1_type, feature_2_type)](data, feature_1, feature_2)

    return


# In[82]:


def visualizations(data):
    """Function for the visualizations page in the web app."""
    st.header("Visualizing our data")
    
column_list = data.columns.to_list()
st.markdown("""
            This section will have visualizations which will be created automatically
            based on rules assigned for the type of variable being visualized. 
            Rules for single variable visualizations:
            * Numerical variables will be represented by histograms.
            * The visualizations for numerical variables will have "Frequency" as the y-axis label.
            * Categorical variables will be represented by bar charts.
            * The visualizations for categorical variables will have "Count" as the y-axis label. 
            """)


st.subheader("Single feature visualization")


# In[84]:


# Dropdown style box to select feature:
feature = st.selectbox(label="Select the feature", options=column_list)
plot_single_feature(data, feature)

st.markdown("""
                Feature interaction visualizations will have two variables
                and will plot the relationship between them.
                Rules for feature interaction visualization:
                * Only two variables can be used for this visualization.
                * Both variables have to be different.
                * For numerical vs numerical visuals, we will be using scatter plots.
                * For numerical vs categorical visuals, we will be using box plots.
                * For categorical vs categorical visuals, we will be using scatter plots.
                """)

st.subheader("Feature interaction visualization")


# Multiselect for selecting two features for interaction plots:
features = st.multiselect(label="Select any two distinct features", 
                              options=column_list)
    # Check for number of features selected:
if len(features) == 2:
    plot_dual_features(data, features[0], features[1])





plot_dual_features(data, "sex", "num_major_vessels")
plot_dual_features(data, "age", "cholesterol")
plot_dual_features(data, "st_slope", "thalassemia")


# In[ ]:


#Data Description

#age: The person's age in years

#sex: The person's sex (1 = male, 0 = female)

#cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)

#trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)

#chol: The person's cholesterol measurement in mg/dl

#fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)

#restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)

#thalach: The person's maximum heart rate achieved

#exang: Exercise induced angina (1 = yes; 0 = no)

#oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)

#slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

#ca: The number of major vessels (0-3)

#thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

#target: Heart disease (0 = no, 1 = yes)


# In[85]:


def create_inference_input(data):
    """
    Function that creates an input form for ML model.
    The function will build the structure for an input form
    using Streamlit functions. The input from the form will be
    taken and converted into a dictionary with the keys being
    the column names of the dataframe and the values being the 
    inputs.
    Parameters
    ----------
    data: DataFrame
        A dataframe containing the the heart disease data.
    
    Returns
    -------
    response_dict: Dict
        A dictionary containing the key, value pairs of the 
        column names of the dataframe and the values from the input
        form.
    """
    input_list = []
    age = st.sidebar.slider(label="Age", 
                            min_value=min(data["age"]), 
                            max_value=max(data["age"]))
    input_list.append(age)
    st.sidebar.write("\n")
    sex = st.sidebar.radio(label="Sex", 
                           options=data["sex"].unique().tolist())
    st.sidebar.write("\n")
    input_list.append(sex)
    chest_pain_type = st.sidebar.selectbox(label="Chest pain type", 
                                           options=data["chest_pain_type"].unique().tolist())
    st.sidebar.write("\n")
    input_list.append(chest_pain_type)
    resting_blood_pressure = st.sidebar.slider(label="Resting blood pressure mm Hg", 
                                               min_value=min(data["resting_blood_pressure"]), 
                                               max_value=max(data["resting_blood_pressure"]))
    st.sidebar.write("\n")
    input_list.append(resting_blood_pressure)
    cholesterol = st.sidebar.slider(label="Cholesterol measurement in mg/dl", 
                                    min_value=min(data["cholesterol"]),
                                    max_value=max(data["cholesterol"]))
    st.sidebar.write("\n")
    input_list.append(cholesterol)
    fasting_blood_sugar = st.sidebar.radio(label="Enter the range for the fasting blood sugar", 
                                           options=data["fasting_blood_sugar"].unique().tolist())
    st.sidebar.write("\n")
    input_list.append(fasting_blood_sugar)
    rest_ecg = st.sidebar.selectbox(label="Resting electromagnetic measurement.", 
                                        options=data["rest_ecg"].unique().tolist())
    st.sidebar.write("\n")
    input_list.append(rest_ecg)
    max_heart_rate_achieved = st.sidebar.slider(label="Maximum heart rate achieved",  
                                                min_value=min(data["max_heart_rate_achieved"]), 
                                                max_value=max(data["max_heart_rate_achieved"]))
    st.sidebar.write("\n")
    input_list.append(max_heart_rate_achieved)
    exercise_induced_angina = st.sidebar.radio(label="Exercise induced Angina?", 
                                               options=data["exercise_induced_angina"].unique().tolist())
    st.sidebar.write("\n")
    input_list.append(exercise_induced_angina)
    st_depression = st.sidebar.slider("Enter the ST depression during exercise", 
                                      min_value=min(data["st_depression"]), 
                                      max_value=max(data["st_depression"]))
    st.sidebar.write("\n")
    input_list.append(st_depression)
    st_slope = st.sidebar.selectbox(label="Slope of peak exercise ST segment", 
                                    options=data["st_slope"].unique().tolist())
    st.sidebar.write("\n")
    input_list.append(st_slope)
    num_major_vessels = st.sidebar.slider(label="Number of major vessels", 
                                          min_value=min(data["num_major_vessels"]), 
                                          max_value=max(data["num_major_vessels"]))
    st.sidebar.write("\n")
    input_list.append(num_major_vessels)
    thalassemia = st.sidebar.selectbox(label="History of Thalassemia?", 
                                       options=data["thalassemia"].unique().tolist())
    st.sidebar.write("\n")
    input_list.append(thalassemia)


# In[86]:


def home(data):
    st.markdown("This is a web application designed to show the CV risks and diseases.")


# In[87]:


def data(data):
    """Function for the data page in web app"""
    st.header("Viewing the dataset")
    # Check dimensions of data:
    st.markdown("""
                The dataset is comprised of {} rows and {} columns including the target variable.
                """.format(data.shape[0], data.shape[1]))
    #TODO(Sayar): Add checkbox here:
    # Read data:
    if st.checkbox("Show data"):
        st.dataframe(data=data)
        st.subheader("Summary Table")
        st.write(df.describe())

    st.subheader("Data Description")
    st.markdown("""
                Here is a bit more fleshed out description of each variable:
                1. **age**: The person's age in years
                2. **sex**: The person's sex (1 = male, 0 = female)
                3. **cp**: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
                4. **trestbps**: The person's resting blood pressure (mm Hg on admission to the hospital)
                5. **chol**: The person's cholesterol measurement in mg/dl
                6. **bs**: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
                7. **restecg**: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
                8. **thalach**: The person's maximum heart rate achieved
                9. **exang**: Exercise induced angina (1 = yes; 0 = no)
                10. **oldpeak**: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot.)
                11. **slope**: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
                12. **ca**: The number of major vessels (0-3)
                13. **thal**: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
                14. **target**: Heart disease (0 = no, 1 = yes)
                """)

    st.subheader("Viewing correlations between the variables")
    st.plotly_chart(make_corr_plot(data))


# In[88]:


st.sidebar.title("Modifaiable risk factors for Heart diseases:")
st.sidebar.markdown("This application is a Share for Top 3 Risk Factors:")


# In[89]:


st.sidebar.title("Risk Factors")
select = st.sidebar.selectbox('Share', ['resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar'], key='1')


# In[ ]:




