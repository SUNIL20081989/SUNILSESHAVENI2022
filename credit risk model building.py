#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


cr=pd.read_csv(r"C:\Users\a980mzz\Downloads\CreditRisk.csv")


# In[3]:


cr.head()


# In[4]:


cr.isnull().sum() # finding values whcih as null for data cleaning purpose


# In[5]:


cr.Credit_History.value_counts()


# In[6]:


#filling bull values
cr.Gender=cr.Gender.fillna('Male')
cr.Married=cr.Married.fillna('Yes')
cr.Dependents=cr.Dependents.fillna(0)
cr.Self_Employed=cr.Self_Employed.fillna('No')
cr.LoanAmount=cr.LoanAmount.fillna(cr.LoanAmount.mean())
cr.Loan_Amount_Term=cr.Loan_Amount_Term.fillna(cr.Loan_Amount_Term.mean())
cr.Credit_History=cr.Credit_History.fillna(1)


# In[7]:


#conveting variables to int
from sklearn.preprocessing  import LabelEncoder
le = LabelEncoder()


# In[8]:


cr.Gender    =  le.fit_transform(cr.Gender)
cr.Married    =  le.fit_transform(cr.Married)
cr.Education =  le.fit_transform(cr.Education)
cr.Self_Employed =  le.fit_transform(cr.Self_Employed)
cr.Property_Area =  le.fit_transform(cr.Property_Area)
cr.Loan_Status =  le.fit_transform(cr.Loan_Status)


# In[9]:


cr=cr.drop(['Loan_ID'],axis=1)
cr.head(1)


# model building
# train & test split

# In[10]:


from sklearn.model_selection  import train_test_split


# In[11]:


cr_train , cr_test  =    train_test_split( cr , test_size= .2)


# In[12]:


cr_train_x = cr_train.iloc[: , 0:-1]
cr_train_y = cr_train.iloc[: , -1]
cr_test_x = cr_test.iloc[: , 0:-1]
cr_test_y = cr_test.iloc[: , -1]


# logistic regression

# In[81]:


from sklearn.linear_model  import LogisticRegression


# In[171]:


cr_logreg=LogisticRegression()


# In[172]:


cr_logreg.fit(cr_train_x,cr_train_y)


# In[173]:


pred_cr_log=cr_logreg.predict(cr_test_x)


# In[174]:


pred_train_cr  =   cr_logreg.predict( cr_train_x )
pred_test_cr   =   cr_logreg.predict( cr_test_x )


# In[175]:


err_train = cr_train_y - pred_train_cr
err_test = cr_test_y  - pred_test_cr


# In[176]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


# In[177]:


tab_cr_log=confusion_matrix(cr_test_y,pred_cr_log)
tab_cr_log


# In[178]:


print("accuracy score with logistic regression -------",accuracy_score(cr_test_y,pred_cr_log)*100)
print("precision score with logistic regression -----",precision_score(cr_test_y,pred_cr_log)*100)
print("F1 score with logistic regression -------",f1_score(cr_test_y,pred_cr_log)*100)


# Decision tree

# In[90]:


from sklearn.tree import DecisionTreeClassifier


# In[91]:


dt_cr=DecisionTreeClassifier()


# In[92]:


dt_cr.fit(cr_train_x,cr_train_y)


# In[93]:


pred_cr_dt=dt_cr.predict(cr_test_x)


# In[94]:


tab_cr_dt=confusion_matrix(cr_test_y,pred_cr_dt)
tab_cr_dt


# In[95]:


print("Accuracy score with decision tree -----", accuracy_score(cr_test_y,pred_cr_dt)*100)
print("precision score with decision tree -----",precision_score(cr_test_y,pred_cr_dt)*100)
print("F1 score with decision tree -------",f1_score(cr_test_y,pred_cr_dt)*100)


# In[96]:


from sklearn.model_selection import GridSearchCV


# In[97]:


search_dt = {'criterion' :['gini', 'entropy']  , 
               'max_depth' :[ 5,6,7,8],
               'min_samples_split'  : [ 50  , 75, 100, 125] }


# finding the best decision tree

# In[98]:


grid_dt = GridSearchCV(dt_cr, param_grid  =search_dt )


# In[99]:


grid_dt.fit(cr_train_x , cr_train_y)


# In[100]:


grid_dt.best_params_  # best decision tree model


# In[101]:


pred_with_best_parm = grid_dt.predict(cr_test_x)
pred_with_best_parm


# In[102]:


Tab_best_dt =confusion_matrix(cr_test_y  ,pred_with_best_parm)
Tab_best_dt


# In[103]:


print("Accuracy score is decision tree-----", accuracy_score(cr_test_y,pred_with_best_parm)*100)
print("precision score with decision tree -----",precision_score(cr_test_y,pred_with_best_parm)*100)
print("F1 score with decision tree -------",f1_score(cr_test_y,pred_with_best_parm)*100)


# In[136]:


from sklearn.model_selection import cross_val_score


# In[137]:


dt_model_cv=cross_val_score(dt_cr,cr_train_x,cr_train_y,cv=5)


# In[138]:


dt_model_cv


# In[139]:


dt_model_cv.mean()


# random forest

# In[143]:


from sklearn.ensemble import RandomForestClassifier
rfc_cr=RandomForestClassifier(class_weight='balanced')


# In[144]:


rfc_cr.fit(cr_train_x,cr_train_y)


# In[145]:


pred_cr_rf=rfc_cr.predict(cr_test_x)


# In[146]:


tab_rfc_cr=confusion_matrix(cr_test_y,pred_cr_rf)
tab_rfc_cr


# In[147]:


print("Accuracy score random forest -----", accuracy_score(cr_test_y,pred_cr_rf)*100)
print("precision score with random forest -----",precision_score(cr_test_y,pred_cr_rf)*100)
print("F1 score with random forest -------",f1_score(cr_test_y,pred_cr_rf)*100)


# finding the best random forest model

# In[156]:


search_randomforest={'criterion' :['gini', 'entropy']  , 
               'n_estimators' :[100,125,150] ,'class_weight':['balanced','None']}


# In[157]:


grid_rf_best = GridSearchCV(rfc_cr, param_grid  = search_randomforest  )


# In[158]:


grid_rf_best.fit(cr_train_x , cr_train_y)


# In[159]:


pred_rf_best=grid_rf_best.predict(cr_test_x)


# In[160]:


grid_rf_best.best_params_


# In[161]:


tab_best_rf=confusion_matrix(cr_test_y,pred_rf_best)
tab_best_rf


# In[162]:


print("Accuracy score is -----", accuracy_score(cr_test_y,pred_rf_best)*100)
print("precision score with random forest -----",precision_score(cr_test_y,pred_rf_best)*100)
print("F1 score with random forest -------",f1_score(cr_test_y,pred_rf_best)*100)


# KNN model

# In[115]:


from sklearn.neighbors import KNeighborsClassifier


# In[116]:


Knn_cr=KNeighborsClassifier(n_neighbors=10)


# In[117]:


Knn_cr.fit(cr_train_x,cr_train_y)


# In[118]:


pred_kn_cr=Knn_cr.predict(cr_test_x)


# In[119]:


tab_KN_cr=confusion_matrix(cr_test_y,pred_kn_cr)
tab_KN_cr


# In[120]:


print("Accuracy score with KNN model-----", accuracy_score(cr_test_y,pred_kn_cr)*100)
print("precision score with KNN model -----",precision_score(cr_test_y,pred_kn_cr)*100)
print("F1 score with KNN model -------",f1_score(cr_test_y,pred_kn_cr)*100)


# In[121]:


from sklearn.neighbors import KNeighborsClassifier
Acc_list=[]
for K in range(1,30):
    Knn_CR= KNeighborsClassifier(n_neighbors=K) 
    Knn_CR.fit(cr_train_x, cr_train_y)
    pred_KNN= Knn_CR.predict(cr_test_x)
    tab_KNN=confusion_matrix(cr_test_y, pred_KNN)
    acc=tab_KNN.diagonal().sum() * 100 / tab_KNN.sum()
    Acc_list.append(acc)


# In[122]:


Acc_list


# In[123]:


plt.plot(Acc_list, marker="*")


# SVM model

# In[124]:


from sklearn.svm import SVC


# In[125]:


cr_svc=SVC()


# In[126]:


cr_svc.fit(cr_train_x,cr_train_y)


# In[127]:


pred_cr_svm=cr_svc.predict(cr_test_x)


# In[128]:


TAB_cr_svm=confusion_matrix(cr_test_y,pred_cr_svm)
TAB_cr_svm


# In[129]:


print("Accuracy score with SVM model-----", accuracy_score(cr_test_y,pred_cr_svm)*100)
print("precision score with SVM model -----",precision_score(cr_test_y,pred_cr_svm)*100)
print("F1 score with SVM model -------",f1_score(cr_test_y,pred_cr_svm)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# naive bytes model

# In[130]:


from sklearn.naive_bayes import MultinomialNB


# In[131]:


NB_cr=MultinomialNB()
NB_cr.fit(cr_train_x,cr_train_y)


# In[132]:


pred_cr_NB=NB_cr.predict(cr_test_x)


# In[133]:


tab_cr_TRF=confusion_matrix(cr_test_y,pred_cr_NB)
tab_cr_TRF


# In[134]:


print("Accuracy score with Naive buyes model  -----", accuracy_score(cr_test_y,pred_cr_NB)*100)
print("precision score with Naive buyes model -----",precision_score(cr_test_y,pred_cr_NB)*100)
print("F1 score with Naive buyes model -------",f1_score(cr_test_y,pred_cr_NB)*100)


# boruta 

# In[72]:


#pip install pandas --trusted-host pypi.org --trusted-host files.pythonhosted.org


# In[ ]:


cr_x=cr.iloc[:,0:11]
cr_y=cr.iloc[:,-1]


# In[35]:


cr_x1=cr_x


# In[36]:


from sklearn.ensemble import RandomForestClassifier


# In[37]:


rf= RandomForestClassifier()


# In[3]:


from boruta import BorutaPy


# In[38]:


cr_x=np.array(cr_x)
cr_y=np.array(cr_y)


# In[39]:


boruta_feat_cr=BorutaPy(rf,max_iter=25,verbose=2)


# In[40]:


boruta_feat_cr.fit(cr_x,cr_y)


# In[24]:


#  at the starting( boruta starts)
# there are 11 x variables ( i do not know which is sig and which is not)
# so at the starting every thing is tentative

# from tentative there can be 3 things( it remains tentaive  or you reject that variable( not significant))


# In[43]:


boruta_feat_cr.support_


# In[42]:


feat_imp=pd.DataFrame({"Feature_names":cr_x1.columns,"IMP":boruta_feat_cr.support_})
feat_imp.sort_values('IMP',ascending=False)


# RFE

# In[56]:


from sklearn.tree import DecisionTreeClassifier


# In[57]:


dt=DecisionTreeClassifier()


# In[58]:


from sklearn.feature_selection import RFE 


# In[64]:


rfe_cr=RFE(dt,n_features_to_select=1)


# In[65]:


rfe_cr.fit(cr_x,cr_y)


# In[66]:


rfe_cr.support_


# In[67]:


feat_imp_rfe=pd.DataFrame({"Feature_names":cr_x.columns,"IMP":rfe_cr.support_})
feat_imp_rfe.sort_values('IMP',ascending=False)


# chisquare

# In[68]:


from scipy.stats import chi2_contingency


# In[70]:


feature_list=[]
score_list=[]
for col in cr_x.columns:
    tab=pd.crosstab(cr_x[col],cr.Loan_Status)
    pvalue=chi2_contingency(tab)[1]
    if pvalue <  .05:
        print("Pvalue is ",pvalue)
        print("feature is significant-----",col)
        print("----------------")
        feature_list.append(col)
        score_list.append(pvalue)
    else:
        print("Pvalue is ",pvalue)
        print("feature is not significant-----",col)
        print("----------------")


# In[71]:


feat_imp_chi=pd.DataFrame({"Feature_list":feature_list,"IMP":score_list})
feat_imp_chi.sort_values('IMP')


# ADABOOST CLASSIFIER

# In[14]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[17]:


dt_cr=DecisionTreeClassifier()
ADA_cr=AdaBoostClassifier(dt_cr,n_estimators=10)


# In[18]:


ADA_cr.fit(cr_train_x,cr_train_y)


# In[20]:


pred_ada_cr=ADA_cr.predict(cr_test_x)


# In[23]:


from sklearn.metrics import confusion_matrix


# In[25]:


tab_ADA_cr=confusion_matrix(cr_test_y,pred_ada_cr)
tab_ADA_cr


# In[ ]:




