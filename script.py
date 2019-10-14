#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import joblib
from matplotlib import pyplot as plt
from sklearn import ensemble 

loan_data = pd.read_csv("loan.csv")

temp_data = loan_data.drop([
  "id",
  'member_id',
  'url',
  'sub_grade',
  'emp_title',
  'sec_app_open_acc',
  'sec_app_revol_util',
  'sec_app_open_act_il',  
  'sec_app_mths_since_last_major_derog',
  'hardship_flag',
  'hardship_type',
  'hardship_reason',
  'hardship_status',
  'deferral_term',
  'hardship_amount',
  'hardship_start_date',
  'hardship_end_date',
  'payment_plan_start_date',
  'hardship_length',
  'hardship_dpd',
  'hardship_loan_status',
  'orig_projected_additional_accrued_interest',
  'hardship_payoff_balance_amount',
  'hardship_last_payment_amount',
  'disbursement_method',
  'debt_settlement_flag',
  'debt_settlement_flag_date',
  'settlement_status',
  'settlement_date',
  'settlement_amount',
  'settlement_percentage',
  'settlement_term',
  'revol_bal_joint',
  'sec_app_earliest_cr_line',
  'sec_app_inq_last_6mths',
  'sec_app_mort_acc',
  'sec_app_num_rev_accts',
  "sec_app_chargeoff_within_12_mths",
  'sec_app_collections_12_mths_ex_med',
  'mths_since_recent_bc_dlq',
  'desc',
  'mths_since_last_delinq',
  'mths_since_last_record',
  'issue_d',
  'title',
  'zip_code',
  'addr_state',
  'earliest_cr_line',
  'initial_list_status',
  'last_pymnt_d',
  'next_pymnt_d',
  'last_credit_pull_d',
  'application_type',
  'verification_status_joint'
],1)

temp_data[:].isnull().sum()

temp_data['emp_length'] = temp_data['emp_length'].replace({
  "1 year": 1,
  '10+ years' : 10, 
  "2 years": 2,
  "3 years": 3,                                 
  "4 years": 4,
  "5 years": 5,
  "6 years": 6,
  "7 years": 7,
  "8 years": 8,
  "9 years": 9, 
  "< 1 year": 1, 
  'NaN': 0 
})

temp_data['term'] = temp_data['term'].replace({
  ' 36 months':36, 
  ' 60 months':60
})

set(temp_data['term'])

temp_data['grade'] = temp_data['grade'].replace({
  "A": 100.00,
  'B' : 80.00, 
  'C': 70.00, 
  'D': 60.00, 
  "E": 50.00, 
  "F": 40.00, 
  "G": 30.00
})

temp_data['home_ownership'] = temp_data['home_ownership'].replace({
  'ANY': 1, 
  'MORTGAGE': 2, 
  'NONE': 0 , 
  'OTHER':3, 
  'OWN':4, 
  'RENT':5
})

set(temp_data['home_ownership'])

temp_data['verification_status'] = temp_data['verification_status'].replace({
  'Not Verified':0, 
  'Source Verified':50, 
  'Verified':100
})

set(temp_data['verification_status'])

temp_data['loan_status'] = temp_data['loan_status'].replace({
  'Charged Off':1,
  'Current':1,
  'Default':0,
  'Does not meet the credit policy. Status:Charged Off':0,
  'Does not meet the credit policy. Status:Fully Paid':0,
  'Fully Paid':1,
  'In Grace Period':1,
  'Late (16-30 days)':1,
  'Late (31-120 days)':0
})

set(temp_data['loan_status'])

temp_data['pymnt_plan'] = temp_data['pymnt_plan'].replace({
  'n':0, 
  'y':1
})

set(temp_data['pymnt_plan'])

temp_data['purpose'] = temp_data['purpose'].replace({
  'car':1,
  'credit_card':2,
  'debt_consolidation':3,
  'educational':4,
  'home_improvement':5,
  'house':6,
  'major_purchase':7,
  'medical':8,
  'moving':9,
  'other':0,
  'renewable_energy':10,
  'small_business':11,
  'vacation':12,
  'wedding':13
})

set(temp_data['purpose'])

list = []
# for i in temp_data:
#     if temp_data[i].dtypes == object:
#         print(temp_data[i])
        #we don't get any objects means removed all the objects and converted some 
        #into float by adding values in the features

model_data = temp_data["mort_acc"].isnull().dropna()
# print(temp_data["mort_acc"].isnull().sum())
# print(model_data.shape)
model_data = temp_data[~ temp_data["mths_since_recent_revol_delinq"].isnull()]
model_data.shape
model_data['avg_cur_bal'].mean()
model_data['emp_length'].fillna(1, inplace = True)
model_data['dti'].fillna(10, inplace = True)
model_data['revol_util'].fillna(60, inplace = True)
model_data['mo_sin_old_il_acct'].fillna(88, inplace = True)
model_data['mo_sin_old_rev_tl_op'].fillna(260, inplace = True)
model_data['mo_sin_rcnt_rev_tl_op'].fillna(10, inplace = True)
model_data['mo_sin_rcnt_tl'].fillna(6, inplace = True)
model_data['mths_since_recent_bc'].fillna(15, inplace = True)
model_data['mths_since_recent_inq'].fillna(7, inplace = True)
model_data['num_accts_ever_120_pd'].fillna(0, inplace = True)
model_data['num_actv_bc_tl'].fillna(6, inplace = True)
model_data['num_actv_rev_tl'].fillna(4, inplace = True)
model_data['num_bc_sats'].fillna(4, inplace = True)
model_data['num_bc_tl'].fillna(7, inplace = True)
model_data['num_il_tl'].fillna(11, inplace = True)
model_data['num_op_rev_tl'].fillna(7, inplace = True)
model_data['num_rev_accts'].fillna(11, inplace = True)
model_data['num_rev_tl_bal_gt_0'].fillna(10, inplace = True)
model_data['num_sats'].fillna(4, inplace = True)
model_data['num_tl_120dpd_2m'].fillna(0, inplace = True)
model_data['num_tl_30dpd'].fillna(0, inplace = True)
model_data['num_tl_90g_dpd_24m'].fillna(0, inplace = True)
model_data['num_tl_op_past_12m'].fillna(3, inplace = True)
model_data['pct_tl_nvr_dlq'].fillna(66, inplace = True)
model_data['percent_bc_gt_75'].fillna(0, inplace = True)
model_data['tot_hi_cred_lim'].fillna(186287.010, inplace = True)
model_data['total_il_high_credit_limit'].fillna(45241, inplace = True)
model_data['policy_code'].fillna(1, inplace = True)
model_data['dti_joint'].fillna(0, inplace = True)
model_data['acc_now_delinq'].fillna(0, inplace = True)
model_data['tot_cur_bal'].fillna(150452, inplace = True)
model_data['open_acc_6m'].fillna(1, inplace = True)
model_data['open_act_il'].fillna(2, inplace = True)
model_data['open_il_12m'].fillna(1, inplace = True)
model_data['open_il_24m'].fillna(1, inplace = True)
model_data['mths_since_rcnt_il'].fillna(21, inplace = True)
model_data['total_bal_il'].fillna(36937, inplace = True)
model_data['il_util'].fillna(69, inplace = True)
model_data['open_rv_12m'].fillna(1, inplace = True)
model_data['open_rv_24m'].fillna(2, inplace = True)
model_data['max_bal_bc'].fillna(5250, inplace = True)
model_data['all_util'].fillna(57, inplace = True)
model_data['total_rev_hi_lim'].fillna(31189, inplace = True)
model_data['inq_fi'].fillna(1, inplace = True)
model_data['total_cu_tl'].fillna(1, inplace = True)
model_data['inq_last_12m'].fillna(2, inplace = True)
model_data['acc_open_past_24mths'].fillna(4.66, inplace = True)
model_data['bc_open_to_buy'].fillna(9488, inplace = True)
model_data['bc_util'].fillna(58, inplace = True)
model_data['chargeoff_within_12_mths'].fillna(0, inplace = True)
model_data['mths_since_last_major_derog'].fillna(23, inplace = True)
model_data['annual_inc_joint'].fillna(6013, inplace = True)
model_data['tot_coll_amt'].fillna(6013, inplace = True)
model_data['avg_cur_bal'].fillna(14203, inplace = True)
model_data['avg_cur_bal'].isnull().sum()
model_data.columns[56]
model_data['chargeoff_within_12_mths'].isnull().sum()
(model_data.shape[0] - model_data.describe().transpose()["count"]).values
set(model_data['loan_status'])

Y = model_data["loan_status"]
X = model_data.drop("loan_status", 1)

# Split X and Y into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# run a simple model
params = {'n_estimators': 3,'max_leaf_nodes':6,'learning_rate': 0.1, 'random_state':1}
classifier = ensemble.GradientBoostingClassifier(**params)
classifier.fit(X_train, Y_train)

# Predict
Y_pred = classifier.predict(X_test)
Y_prob = classifier.predict_proba(X_test)

# Make the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
# print(cm)

# Calculate AUC 
from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_test, Y_pred))

# Applying Grid Search to find the best model with the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{
  'n_estimators': [1000],
  'max_leaf_nodes': [10],
  'learning_rate': [0.1],
  'random_state': [1]
}]
grid_search = GridSearchCV(
  estimator=classifier,
  param_grid=parameters,
  scoring="accuracy",
  cv=10,
  n_jobs=-1
)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy is: ", best_accuracy)
print("Best Parameters are: ", best_parameters)