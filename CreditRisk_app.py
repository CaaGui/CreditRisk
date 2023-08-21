import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("Paired")


#from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier





# Set no tema do seaborn para melhorar o visual dos plots
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


@st.cache(show_spinner= True, allow_output_mutation=True)
def load_data(file_data):
    try:
        return pd.read_csv(file_data)
    except:
        return pd.read_excel(file_data)



def get_model():
    clf = GradientBoostingClassifier(random_state=100,n_estimators= 600,    
                                    min_samples_leaf = 2,
                                    learning_rate=.1
                                    )
            
    return clf



# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'Credit Risk', \
        page_icon = '',
        layout="wide",
        initial_sidebar_state='expanded'
    )

    st.markdown(''' ## Introduction

    #### Context
    The original dataset contains 32581 entries with 12 categorial variables .
    In this dataset, each entry represents a person who takes a credit by a bank. 
    Each person is classified as good or bad credit risks according to the set of variables.
    The link to the original dataset can be found below:
            
    https://www.kaggle.com/datasets/laotse/credit-risk-dataset')


    #### Content
    It is almost impossible to understand the original dataset due to its
    complicated system of categories and symbols. Thus, I wrote a small 
    Python script to convert it into a readable CSV file. 
    Several columns are simply ignored, because in my opinion 
    either they are not important or their descriptions are obscure. The selected attributes are:
    


    | Feature Name         |                  Description                        | Type  |
    | -------------------- |:---------------------------------------------------:| -----:|
    | person_age    |                Age            |        integer             |
    | person_income |             Annual income          |                integer                   |
    | person_home_ownership |            Home ownership           |              text                  |
    | person_emp_length |         Employment, length (in years)        |        float           |
    | loan_intent |          Loan intent                   |              text                     |
    | loan_grade |               Loan grade                |                 text                      |
    | loan_amnt |                Loan amount                 |                   integer                    |
    | loan_int_rate |              Interest rate              |                   float               |
    | loan_percent_income |               Percent income            |                  float      |
    | cb_person_default_on_file |        Historical default             |            binary            |
    | cb_preson_cred_hist_length |       Credit history length          |           integer            |
    | **loan_status** |          Loan status (0 is non default 1 is default)  |     binary                  |



    ### **LOADING DATA**

    The CSV file containing the data is located in same place of this notebook

    ''')

    df_o = load_data("./credit_risk_dataset.csv")

    st.write('Data shape, lines and columns:', df_o.shape)
    
    st.write(df_o.info())

    st.table(df_o.head())








###                 UNIVARIATE
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################

    st.markdown('## Univariate')

    var = 'loan_status'    
    st.write(f'Variable : {var}')
    fig , ax = plt.subplots(figsize=(8,1))
    df_o[var].value_counts().plot.barh(ax=ax)
    st.pyplot(fig)


    var = 'person_age'
    st.write(f'Variable : {var}')
    fig , ax = plt.subplots(1,2, figsize=(8,3))
    pd.qcut(df_o[var],3, precision=0).value_counts().sort_index().plot.bar(ax=ax[0])
    df_o[var].plot.box(ax=ax[1])
    st.pyplot(fig)


    var = 'person_income'
    st.write(f'Variable : {var}')
    fig , ax = plt.subplots(1,2, figsize=(8,3))
    pd.qcut(df_o[var],3, precision=0).value_counts().sort_index().plot.bar(ax=ax[0])
    df_o[var].plot.box(ax=ax[1])
    st.pyplot(fig)


    var = 'person_home_ownership'
    st.write(f'Variable : {var}')
    st.write(df_o[var].value_counts())
    fig , ax = plt.subplots(figsize=(8,2))
    df_o[var].value_counts().plot.barh(ax= ax)
    st.pyplot(fig)


    var = 'person_emp_length'
    st.write(f'Variable : {var}')
    fig , ax = plt.subplots(1,2, figsize=(8,3))
    pd.qcut(df_o[var],4, precision=0).value_counts().sort_index().plot.bar(ax=ax[0])
    df_o[var].plot.box(ax=ax[1])
    st.pyplot(fig)


    var = 'loan_intent'
    st.write(f'Variable : {var}')
    print(df_o[var].value_counts())
    fig , ax = plt.subplots(figsize=(8,2))
    df_o[var].value_counts().plot.barh(ax=ax)
    st.pyplot(fig)


    var = 'loan_grade'
    st.write(f'Variable : {var}')
    st.write(df_o[var].value_counts())
    fig , ax = plt.subplots(figsize=(8,2))
    df_o[var].value_counts().sort_index(ascending=False).plot.barh(ax= ax)
    # chart.tick_params(axis='x', labelrotation = 0)
    st.pyplot(fig)


    var = 'loan_amnt'
    st.write(f'Variable : {var}')
    fig , ax = plt.subplots(1,2, figsize=(8,3))
    pd.qcut(df_o[var],3, precision=0).value_counts().sort_index().plot.bar(ax=ax[0])
    df_o[var].plot.box(ax=ax[1])
    st.pyplot(fig)


    var = 'loan_int_rate'
    st.write(f'Variable : {var}')
    fig , ax = plt.subplots(1,2, figsize=(8,3))
    pd.qcut(df_o[var],5, precision=3).value_counts().sort_index().plot.bar(ax=ax[0])
    df_o[var].plot.box(ax=ax[1])
    st.pyplot(fig)


    var = 'loan_percent_income'
    st.write(f'Variable : {var}')
    fig , ax = plt.subplots(1,2, figsize=(8,3))
    pd.qcut(df_o[var],5, precision=3).value_counts().sort_index().plot.bar(ax=ax[0])
    df_o[var].plot.box(ax=ax[1])
    st.pyplot(fig)


    var = 'cb_person_default_on_file'
    st.write(f'Variable : {var}')
    st.write(df_o[var].value_counts())
    fig , ax = plt.subplots(figsize=(8,3))
    df_o[var].value_counts().plot.barh(ax= ax)
    st.pyplot(fig)


    var = 'cb_person_cred_hist_length'
    st.write(f'Variable : {var}')
    fig , ax = plt.subplots(1,2, figsize=(8,3))
    pd.qcut(df_o[var],3, precision=0).value_counts().sort_index().plot.bar(ax=ax[0])
    df_o[var].plot.box(ax=ax[1])
    st.pyplot(fig)







###                 BIVARIATE
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
    st.markdown('## Bivariate')
    var = 'person_age'
    st.write(f'Variable loan_status related to {var}')
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.kdeplot(data= df_o, hue='loan_status', x=var, ax= ax[0])
    sns.boxplot(data= df_o, x='loan_status', y=var, ax= ax[1])
    st.pyplot(fig)


    var = 'person_income'
    st.write(f'Variable loan_status related to {var}')
    fig, ax = plt.subplots(2, figsize=(12,5))
    sns.kdeplot(data= df_o, hue='loan_status', x= var, ax= ax[0])
    sns.boxplot(data= df_o, y='loan_status', x= var, orient='h', ax= ax[1])
    st.pyplot(fig)


    var = 'person_home_ownership'
    col1, col2 = st.columns(2)
    col1.write(f'Variable loan_status related to {var}')
    col2.write(df_o.groupby(var)['loan_status'].value_counts())
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    df_o[df_o['loan_status']==0][var].value_counts().plot.pie(ax=ax[0],autopct= "%1.0f%%",startangle=140,
                                                            wedgeprops={"linewidth":1,"edgecolor":"white"},
                                                            explode=[.02,.02,.0,.0], title='loan_status=0');
    df_o[df_o['loan_status']==1][var].value_counts().plot.pie(ax=ax[1],autopct= "%1.0f%%",startangle=120,
                                                            wedgeprops={"linewidth":1,"edgecolor":"white"},
                                                            explode=[.02,.02,.0,.15], title='loan_status=1');
    sns.displot(data= df_o, y=var,  col='loan_status')    
    st.pyplot(fig)


    var = 'person_emp_length'
    st.write(f'Variable loan_status related to {var}')
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.kdeplot(data= df_o, hue='loan_status', x= var, ax= ax[0])
    sns.boxplot(data= df_o, x='loan_status', y= var, ax= ax[1])    
    st.pyplot(fig)


    var = 'loan_intent'
    col1, col2 = st.columns(2)
    col1.write(f'Variable loan_status related to {var}')
    col2.write(df_o.groupby(var)['loan_status'].value_counts())
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    df_o[df_o['loan_status']==0][var].value_counts().plot.pie(ax=ax[0],autopct= "%1.0f%%",startangle=120,
                                                            wedgeprops={"linewidth":1,"edgecolor":"white"},
                                                            explode=[.03,.03,.0,.0,.0,.0], title='loan_status=0');
    df_o[df_o['loan_status']==1][var].value_counts().plot.pie(ax=ax[1],autopct= "%1.0f%%",startangle=120,
                                                            wedgeprops={"linewidth":1,"edgecolor":"white"},
                                                            explode=[.03,.03,.0,.0,.0,.0], title='loan_status=1');
    sns.displot(data= df_o, y=var,  col='loan_status')
    st.pyplot(fig)


    var = 'loan_grade'
    col1, col2 = st.columns(2)
    col1.write(f'Variable loan_status related to {var}')
    col2.write(df_o.groupby(var)['loan_status'].value_counts())
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    df_o[df_o['loan_status']==0][var].value_counts().plot.pie(ax=ax[0],autopct= "%1.0f%%",startangle=120,
                                                            wedgeprops={"linewidth":1,"edgecolor":"white"},
                                                            explode=[.05,.05,.0,.0,.0,.15,.3], title='loan_status=0');
    df_o[df_o['loan_status']==1][var].value_counts().plot.pie(ax=ax[1],autopct= "%1.0f%%",startangle=120,
                                                            wedgeprops={"linewidth":1,"edgecolor":"white"},
                                                            explode=[.05,.05,.0,.0,.0,.15,.3], title='loan_status=1');
    sns.displot(data= df_o, x=var,  col='loan_status')
    st.pyplot(fig)


    var = 'loan_amnt'
    st.write(f'Variable loan_status related to {var}')
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.kdeplot(data= df_o, hue='loan_status', x= var, ax= ax[0])
    sns.boxplot(data= df_o, x='loan_status', y= var, ax= ax[1])
    st.pyplot(fig)


    var = 'loan_int_rate'
    st.write(f'Variable loan_status related to {var}')
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.kdeplot(data= df_o, hue='loan_status', x= var, ax= ax[0])
    sns.boxplot(data= df_o, x='loan_status', y= var, ax= ax[1])
    st.pyplot(fig)


    var = 'loan_percent_income'
    st.write(f'Variable loan_status related to {var}')
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.kdeplot(data= df_o, hue='loan_status', x= var, ax= ax[0])
    sns.boxplot(data= df_o, x='loan_status', y= var, ax= ax[1])
    st.pyplot(fig)


    var = 'cb_person_default_on_file'
    col1, col2 = st.columns(2)
    col1.write(f'Variable loan_status related to {var}')
    col2.write(df_o.groupby(var)['loan_status'].value_counts())
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    df_o[df_o['loan_status']==0][var].value_counts().plot.pie(ax=ax[0],autopct = "%1.0f%%",startangle=90,
                                                            wedgeprops={"linewidth":1,"edgecolor":"white"},
                                                            explode=[.05,.0], title='loan_status=0');
    df_o[df_o['loan_status']==1][var].value_counts().plot.pie(ax=ax[1],autopct = "%1.0f%%",startangle=120,
                                                            wedgeprops={"linewidth":1,"edgecolor":"white"},
                                                            explode=[.05,.0], title='loan_status=1');
    sns.displot(data= df_o, y=var,  col='loan_status')
    st.pyplot(fig)


    var = 'cb_person_cred_hist_length'
    st.write(f'Variable loan_status related to {var}')
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.kdeplot(data= df_o, hue='loan_status', x= var, ax= ax[0])
    sns.boxplot(data= df_o, x='loan_status', y= var, ax= ax[1])
    st.pyplot(fig)






###                 DATA STRUCTURE
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
    st.markdown('## Data Structuring & Processing')

    col1, col2 = st.columns(2)
    col1.write('Missing Data %')
    z = (df_o.isna().sum() / df_o.shape[0])
    col1.write(z.apply(lambda x: '{:.2%}'.format(x)))
    col2.write('\nLines with missing values: {:.2%}'.format(df_o.isna().sum().sum() / df_o.shape[0]))


    df = df_o.dropna().copy()
    print()
    st.write('Data shape, lines and columns:', df.shape)
    print()

    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].replace({'Y':1,'N':0})

    df = pd.get_dummies(df)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="RdYlGn",linewidth =1)
    st.pyplot(fig)

    X = df.drop(columns='loan_status')
    y = df['loan_status']






###                 DATA ANALYSIS
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
###############     ###############    ############   ###########     #######################
    st.markdown('## Assesment')

    filt = (df_o['person_emp_length'].isna()==1) | (df_o['loan_int_rate'].isna()==1)
    
    st.write('''
    The total loan amount in the data : $ {:,.0f}

    The average loan amount in the data : $ {:,.2f}


    The variable, person_emp_length, which lines is missing values, represents a loan amount of :
    - Sum: $ {:,.0f}
    - Mean: $ {:,.2f}

    The variable, loan_int_rate, which lines is missing values, represents a loan amount of :
    - Sum: $ {:,.0f}
    - Mean: $ {:,.2f}

    The total loan amount of both variables with missing values is :
    $ {:,.0f}

    Was decided to remove lines containing missing values. The amount it represents is lower than 15%
    and some fundamental informations as the interest rate applied in the loans is crucial to define the 
    value of risk for each client as to understand if the loan could be given. So, modeling the data with 
    no interest rates for the given loan doesnt have any sense and will jeopardize the success of the model.
    '''.format(
                df_o.loc[:,'loan_amnt'].sum(),
                df_o.loc[:,'loan_amnt'].mean(),
                df_o.loc[df_o['person_emp_length'].isna()==1,'loan_amnt'].sum(),
                df_o.loc[df_o['person_emp_length'].isna()==1,'loan_amnt'].mean(),
                df_o.loc[df_o['loan_int_rate'].isna()==1,'loan_amnt'].sum(),
                df_o.loc[df_o['loan_int_rate'].isna()==1,'loan_amnt'].mean(),
                df_o.loc[filt,'loan_amnt'].sum()
            )
    )

    st.write('''

    Before removing missing values
        Approved loans (0 is denied and 1 is Approved):
    ''')

    col1, col2 = st.columns(2)
    col2.write(df_o['loan_status'].value_counts())
    col1.write('''
        Rate of bad loans : {:.2%}
    '''.format(df_o['loan_status'].mean())
    )

    st.write('''

    After removing missing values
        Approved loans (0 is denied and 1 is Approved):
    ''')
    
    col1, col2 = st.columns(2)
    col2.write(df['loan_status'].value_counts()) 
    col1.write('''    
        Rate of bad loans  : {:.2%}
    '''.format(df['loan_status'].mean())
    )

    st.write('''


    Even after removing missing values the rate of bad loans kept similar.

    The model then achieved a precision of .9345 or 93,45% of accuray.

    Other variables and informations in wich could be classified as outliers was maintained as a way to
    keep the model flexible enablening a higher variance and not reaching an overfitting.


    ''')






if __name__ == '__main__':
	main()