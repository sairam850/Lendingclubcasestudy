import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import re

# Ignore warnings due to version problems
import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.2f}'.format 

# Loading the Lending Club Case Study

input = pd.read_csv("C://Users//Sairam//Downloads//loan//loan.csv",dtype=object)
input.head()

input.shape

input.info()

# Investigating Null Values

input.isnull().sum()

input.isnull().all(axis = 0).sum()

input.isnull().all(axis = 1).sum()

# Data Cleaning

input.drop(input.iloc[:, 53:105],inplace = True, axis  = 1)

input.head()

column1 = ["desc","mths_since_last_delinq","mths_since_last_record","next_pymnt_d","tot_hi_cred_lim"]
input.drop(labels = column1, axis =1, inplace=True)





column2 = ["mths_since_last_major_derog","total_bal_ex_mort","total_bc_limit","total_il_high_credit_limit"]
input.drop(labels = column2, axis =1, inplace=True)

column3 = ["member_id","url","emp_title","zip_code","tax_liens"]
input.drop(labels = column3, axis =1, inplace=True)

input.shape

input.isnull().sum()

column4 = ["application_type","policy_code","initial_list_status","installment","pymnt_plan"]
input.drop(labels = column4, axis =1, inplace=True)

input.head()

input.emp_length.fillna('0',inplace = True)
input["emp_length"] = input.emp_length.str.extract('(\d+)')
input.head(2)

print(input.pub_rec_bankruptcies.isnull().sum())

input.pub_rec_bankruptcies.fillna('Not Known',inplace = True)
print(input.pub_rec_bankruptcies.isnull().sum())

input['int_rate'] = input['int_rate'].str.rstrip('%')

input['revol_util'] = input['revol_util'].str.rstrip('%')

columns_Numeric = ['loan_amnt','funded_amnt','funded_amnt_inv','int_rate','emp_length','annual_inc','dti']
input[columns_Numeric] = input[columns_Numeric].apply(pd.to_numeric)

(input.loan_status.value_counts()*100)/len(input)

(input.purpose.value_counts()*100)/len(input)

input.issue_d = pd.to_datetime(input.issue_d, format='%b-%y')
input['year'] = input ['issue_d'].dt.year
input ['month'] = input ['issue_d'].dt.month
input.head(5)

# UNIVARIATE ANALYSIS

input['loan_amnt'].describe()

sns.boxplot(input['loan_amnt'])

input['total_pymnt'].describe()



# Outlier Treatment

# Before Removing Outliers

print(input['annual_inc'].describe())

sns.boxplot(input['annual_inc'])

# After Removing Outliers

input = input[input['annual_inc']<input['annual_inc'].quantile(0.95)]

print(input['annual_inc'].describe())

sns.boxplot(input['annual_inc'])

input['int_rate'].describe()

sns.boxplot(input['int_rate'])

# Distribution Plot

sns.distplot(input['int_rate'])

sns.distplot(input['annual_inc'])

sns.distplot(input['total_pymnt'])

loan_correlation = input.corr()
sns.heatmap(loan_correlation,annot = True,cmap="BrBG")
plt.show()

sns.clustermap(loan_correlation,annot = True,cmap="BrBG")
plt.show()

input['loan_amnt_cat'] = pd.cut(input['loan_amnt'],[0,5000,10000,15000,20000,25000,30000,35000],labels = ['0-5000','5000-10000','10000-15000','15000-20000','20000-25000','25000-30000','30000+'])

input['annual_inc_cat'] = pd.cut(input['annual_inc'],[0,15000,30000,45000,60000,75000,90000,105000],labels = ['0-15000','15000-30000','30000-45000','45000-60000','60000-75000','75000-90000','90000+'])

input['int_rate_cat'] = pd.cut(input['int_rate'],[5,10,12.5,16,20],labels = ['5-10','10-12.5','12.5-16','16+'])

input['dti_cat'] = pd.cut(input['dti'],[0,5,10,15,20,25],labels = ['0-5','5-10','10-15','15-20','20+'])

input['loan_amnt_cat'].describe()

plt.figure(figsize=(14,8),facecolor='w')
plt.subplot(2,2,1)
ax = sns.distplot(input['annual_inc'],rug =True)
ax.set_title("Annual Income Distribution plot",fontsize=15,color='g')
ax.set_xlabel("Annual Income",fontsize=14,color='g')

plt.subplot(2,2,2)
ax = sns.boxplot(y = input['annual_inc'])
ax.set_title("Annual Income Box plot",fontsize=15,color='g')
ax.set_xlabel("Annual Income",fontsize=14,color='g')

plt.show()

# Univariate Analysis Categorical Variables

plt.figure(figsize=(14,8),facecolor='w')
ax = sns.countplot(y = "purpose",data = input , hue = "loan_status",palette = "GnBu_d")
ax.legend(bbox_to_anchor=(1, 1))
ax.set_title("purpose",fontsize=15,color='g')

ax.set_xlabel("Loan_Application_Count",fontsize=14,color='g')
ax.set_ylabel("purpose",fontsize=14,color='g')

plt.show()

# Segmented Univariate Analysis

plt.figure(figsize=(14,8),facecolor='w')


input.groupby(['year','month']).id.count().plot(kind = 'bar')
ax.set_xlabel("Year,Month",fontsize=14,color='g')
ax.set_ylabel("Loan application Count",fontsize=14,color='g')
ax.set_title("Loan application count in years and months",fontsize=15,color='g')

plt.show()

plt.figure(figsize=(14,8),facecolor='w')
ax = sns.countplot(x = "home_ownership",data = input , hue = "loan_status",palette = "GnBu_d")
ax.legend(bbox_to_anchor=(1, 1))
ax.set_title("loan_status",fontsize=15,color='g')

ax.set_xlabel("home_ownership",fontsize=14,color='g')
ax.set_ylabel("Loan_Application_Count",fontsize=14,color='g')

plt.show()

plt.figure(figsize=(14,8),facecolor='w')
ax = sns.countplot(x = "term",data = input , hue = "loan_status",palette = "GnBu_d")
ax.legend(bbox_to_anchor=(1, 1))
ax.set_title("Term_Loan Status",fontsize=15,color='g')

ax.set_xlabel("Loan Term",fontsize=14,color='g')
ax.set_ylabel("Loan_Application_Count",fontsize=14,color='g')

plt.show()

# Bivariate Analysis

interest_vs_loan = input.groupby(['int_rate_cat', 'loan_status']).loan_status.count().unstack().fillna(0).reset_index()
interest_vs_loan['Total'] = interest_vs_loan['Charged Off'] + interest_vs_loan['Current'] + interest_vs_loan['Fully Paid'] 
interest_vs_loan['Chargedoff_Proportion'] = interest_vs_loan['Charged Off'] / interest_vs_loan['Total']
interest_vs_loan.sort_values('Chargedoff_Proportion', ascending=False)

fig,ax1 =  plt.subplots(figsize=(14,8),facecolor='w')
ax1.set_title("Term_Loan Status",fontsize=15,color='g')
ax1 = sns.barplot(x = 'int_rate_cat', y = 'Chargedoff_Proportion',data = interest_vs_loan )
ax1.set_xlabel("Interest Rates",fontsize=14,color='g')
ax1.set_ylabel("Chargedoff_Proportion",fontsize=14,color='g')

plt.show()

plt.figure(figsize=(14,8),facecolor='w')
ax = sns.boxplot(x='int_rate', y='purpose', data =input,palette='rainbow')
ax.set_title('Purpose vs Interest Rate',fontsize=15,color='g')
ax.set_xlabel('Interest Rate',fontsize=14,color = 'g')
ax.set_ylabel('Purpose of Loan',fontsize=14,color = 'g')
plt.show()

plt.figure(figsize=(14,8),facecolor='w')
ax = sns.boxplot(y='int_rate', x='year', data =input,palette='gnuplot')
ax.set_title('Year vs Interest rates',fontsize=15,color='g')
ax.set_xlabel('Year',fontsize=14,color = 'g')
ax.set_ylabel('Interest Rate',fontsize=14,color = 'g')
plt.show()







