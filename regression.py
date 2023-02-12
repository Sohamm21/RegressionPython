pip install scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Regression Dashboard")

file_type = st.radio("Select the type of file:", 
                                 ["CSV", "XLS"])

file_uploaded = 0
if file_type == "CSV":
    st.write("***Upload your dataset in csv format:***")
    file = st.file_uploader("Choose a CSV file", type="csv")
elif file_type == "XLS":
    st.write("Upload your dataset in XLS format:")
    file = st.file_uploader("Choose a XLS file", type="xls")

if (file is not None) and (file_type == "CSV"):
    data = pd.read_csv(file)
    st.dataframe(data.head())
    file_uploaded = 1
elif (file is not None) and (file_type == "XLS"):
    data = pd.read_excel(file)
    st.dataframe(data.head())
    file_uploaded = 1
    
if file_uploaded == 1:
    reg_type = st.selectbox("Regression type: ",
                     ['Linear', 'Logistic', 'Multivariate'])
    st.write("You selected:", reg_type)
    if (reg_type == "Linear"):
        st.write("Select the column to be used as the independent variable (X):")
        independent_var = st.selectbox("", data.columns, key='unique_key_1')

        st.write("Select the column to be used as the dependent variable (Y):")
        dependent_var = st.selectbox("", data.columns, key='unique_key_2')

        x = data[[independent_var]]
        y = data[dependent_var]

        st.write("Split the dataset into training and testing sets:")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        st.write("Fit the linear regression model:")
        model = LinearRegression()
        model.fit(x_train, y_train)

        st.write("Predict the values for the test set:")
        y_pred = model.predict(x_test)

        st.write("Evaluate the model using mean absolute error and mean squared error:")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write("Mean Absolute Error:", mae)
        st.write("Mean Squared Error:", mse)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        st.write("Root Mean Squared Error:", rmse)
        st.write("R^2 score:", r2)
        m = model.coef_[0][0]
        c = model.intercept_[0]
        st.write("Slope (m):", m)
        st.write("Y-intercept (c):", c)
        st.write("Plot the regression line:")
        plt.scatter(x_test, y_test, color='gray')
        plt.plot(x_test, y_pred, color='red', linewidth=2)
        plt.title("Linear Regression")
        plt.xlabel(independent_var)
        plt.ylabel(dependent_var)
        st.pyplot()
    
    elif (reg_type == "Multivariate"):
        independent_vars = st.multiselect("Select independent variables", data.columns.tolist(), default=[data.columns[1]])
        X = data[independent_vars]
        dependent_var = st.selectbox("Select dependent variable", data.columns.tolist(),index=data.columns.tolist().index(data.columns[-1]))
        y = data[dependent_var]
        reg = linear_model.LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        st.write("Mean Absolute Error:", mae)
        st.write("Mean Squared Error:", mse)
        st.write("R2 score:", r2)
        st.write("Root Mean Square Error:", rmse)
        
        bed = st.number_input("Total bedrooms: ", value=0, step=1)
        bath = st.number_input("Total bathrooms: ", value=0.0, step=0.1)
        sqliv = st.number_input("Sqft living: ", value=0, step=1)
        sqlot = st.number_input("Sqft lot: ", value=0, step=1)
        floors = st.number_input("Total floors: ", value=0.0, step=0.1)
        view = st.number_input("View: ", value=0, step=1)
        waterfront = st.number_input("Waterfront: ", value=0.0, step=0.1)
        condition = st.number_input("Condition: ", value=0, step=1)
        sqftAbv = st.number_input("Sqft above: ", value=0, step=1)
        sqftBase = st.number_input("Sqft basement: ", value=0, step=1)
        yrBuilt = st.number_input("Year Built: ", value=0, step=1)
        yrReno = st.number_input("Year renovated: ", value=0, step=1)
        instance = pd.DataFrame([[bed, bath, sqliv, sqlot, floors, waterfront, view, condition, sqftAbv, sqftBase, yrBuilt, yrReno]], columns=["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'])
        predict = reg.predict(instance)
        st.write("The house price is: ", predict[0])
        
    elif (reg_type == "Logistic"):
        data = data.drop(columns = ['Id'])
        le = LabelEncoder()
        data['Species'] = le.fit_transform(data['Species'])
        
        X = data.drop(columns = ['Species'])
        Y = data['Species']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

        model = LogisticRegression()
        model.fit(X_train, Y_train)        
        st.write("Accuracy:", model.score(X_test, Y_test) * 100)
        
        sepLen = st.number_input("Enter the sepal length:", value=0.0, step=0.1)
        sepWidth = st.number_input("Enter the sepal width:", value=0.0, step=0.1)
        petLen = st.number_input("Enter the petal length:", value=0.0, step=0.1)
        petWidth = st.number_input("Enter the petal width:", value=0.0, step=0.1)
        instance = pd.DataFrame([[sepLen, sepWidth, petLen, petWidth]], columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
        prediction = model.predict(instance)
        if prediction[0] == 0:
            st.write("The species is: Iris-setosa")
        elif prediction[0] == 1:
            st.write("The species is: Iris-versicolor")
        else:
            st.write("The species is: Iris-virginica")
        
        


    
