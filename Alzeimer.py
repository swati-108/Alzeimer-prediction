from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
ds=pd.read_csv('/content/drive/MyDrive/spt/GSE33000_final.csv')


features_name=ds.columns
features_name = features_name[6:]
print(features_name.shape)

from sklearn.impute import SimpleImputer
X=ds[features_name].to_numpy()
y=ds['Disease Status'].to_numpy()
for i in range(0,624):
  if y[i]==2:
    y[i]=0
print(X)
ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)


from sklearn.feature_selection import SelectKBest, chi2

K=40
score_chi2=[]
ID_chi=[]

ch2 = SelectKBest(chi2, k=K)
X1= ch2.fit_transform(X, y)

nn1= ch2.get_support()
F1_len=len(nn1)

for i in range ( F1_len):
        if nn1[i]==True:
            
            print( features_name[i],":",ch2.scores_[i])
            
            score_chi2.append(ch2.scores_[i])
            #ID_chi.append(features_Id[i])
    # # print( ch2.get_params())
    # ##--------------------------------------------
ID_chi=[int(i) for i in ID_chi] # to convert list from float to int 
m_ID_chi2 = []
for j in range(len(ID_chi)):
    m_ID_chi2.append([ID_chi[j], score_chi2[j]])
m_ID_chi2.sort(key=lambda x: x[1], reverse=True)

from sklearn.feature_selection import f_classif


K=20
score_anova=[]
ID_anova=[]
Anova_fs = SelectKBest(f_classif, k=K)
X2= Anova_fs.fit_transform(X, y)
nn2= Anova_fs.get_support()
F2_len=len(nn2)

for i in range ( F2_len):
        if nn2[i]==True:
            
            print( features_name[i],":",ch2.scores_[i])
            # names_anova.append(features_name[i])
            #score_anova.append(Anova_fs.scores_[i])
            #ID_anova.append(features_Id[i])
  
ID_anova=[int(i) for i in ID_anova] # to convert list from float to int 
m_ID_anova = []
for j in range(len(ID_anova)):
    m_ID_anova.append([ID_anova[j], score_anova[j]])
m_ID_anova.sort(key=lambda x: x[1], reverse=True)

from sklearn.feature_selection import mutual_info_classif

K=20
score_MI=[]
ID_MI=[]
MI_fs = SelectKBest(mutual_info_classif, k=10)
X3= MI_fs.fit_transform(X, y)
nn3= MI_fs.get_support()
F3_len=len(nn3)

for i in range ( F3_len):
        if nn3[i]==True:
            
            print( features_name[i],":",ch2.scores_[i])
            # names_MI.append(features_name[i])
            #ID_MI.append(features_Id[i])
            #score_MI.append(MI_fs.scores_[i])
  
    # ##--------------------------------------------
ID_MI=[int(i) for i in ID_MI] # to convert list from float to int 
m_ID_MI = []
for j in range(len(ID_MI)):
    m_ID_MI.append([ID_MI[j], score_MI[j]])
m_ID_MI.sort(key=lambda x: x[1], reverse=True)


X=ds[['OLFM1,AMY,NOE1,OlfA,NOELIN,NOELIN1,NOELIN1_V1,NOELIN1_V2,NOELIN1_V4','CRH,CRF']].to_numpy()

ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)

from sklearn.model_selection import train_test_split
X_tr, X_ts, y_tr, y_ts= train_test_split(X, y, test_size=125, random_state=44, shuffle =True)

print('X_train shape is ' , X_tr.shape)
print('X_test shape is ' , X_ts.shape)
print('y_train shape is ' , y_tr.shape)
print('y_test shape is ' , y_ts.shape)

from sklearn.model_selection import RepeatedStratifiedKFold
k=10
n=30
Id_case=[]

# acc_arr_A8=[]
# acc_arr_S8=[]
# acc_arr_R8=[]
# acc_arr_L8=[]
rskf = RepeatedStratifiedKFold(n_splits=k,n_repeats=n,random_state=44)
rskf.get_n_splits(X_tr, y_tr)

precision_av=0
Recall_av=0
specificity_av=0
acc_av=0
f1_av=0
roc_auc_av=0
cm_av=0
kappa_av=0

    
for train_index1, test_index1 in rskf.split(X_tr, y_tr):
        X_train1, X_test1 =X_tr[[train_index1]],X_tr[[test_index1]]
        y_train1, y_test1 = y_tr[train_index1],y_tr[test_index1]
  
#print('X_train1' , X_train1.shape)
print(X_train1[0])
case=X_train1[0]
Id_case.append(case)


print(X_train1.shape)


from sklearn.svm import SVC
SVM = SVC(kernel="linear")

SVM.fit(X_train1, y_train1)
fit_time = time.time()- t0
        % fit_time)
        
train_score= SVM.score(X_train1, y_train1)
#        print('SVM Train Score is : ' ,train_score )
        
test_score= SVM.score(X_test1, y_test1)
#        print('SVM Test Score is : ' , test_score )
        
y_pred = SVM.predict(X_test1)

from sklearn import metrics
from sklearn.metrics import *

cm = confusion_matrix(y_test1, y_pred) 
print('Confusion Matrix : \n', cm)
cm_av+=cm
acc_arr_S8=[]        
      
accuracy_score=metrics.accuracy_score(y_test1, y_pred)
acc_arr_S8.append(accuracy_score)
print("accuracy_score=",accuracy_score)
acc_av+=accuracy_score
        
################################# f1_score
from sklearn.metrics import f1_score
f1_sco=f1_score(y_test1, y_pred)
print('f1_score=',f1_sco)        
f1_av+=f1_sco
        
########################### roc_auc_score  
roc_auc = metrics.roc_auc_score(y_test1,y_pred)
print("roc_auc=",roc_auc)
roc_auc_av+=roc_auc

            
################ precision
PrecisionScore = precision_score(y_test1, y_pred)
print("Precision:",PrecisionScore)
precision_av+=PrecisionScore
        
################Recall       
Rec=recall_score(y_test1, y_pred)
print("recall:", Rec)
Recall_av+=Rec
        
###################specificity
spec = cm[1,1]/(cm[1,1]+cm[1,0])
print('Specificity : ', spec)
specificity_av+=spec
        
############# kappa
kappa=cohen_kappa_score(y_test1, y_pred)
print('kappa:', kappa)
kappa_av+=kappa
         
######### get the average of the evaluation metrics            

precision_average=(precision_av/n)/k
Recall_average=(Recall_av/n)/k
spec_average=(specificity_av/n)/k
kappa_average=(kappa_av/n)/k
acc_average= (acc_av/n)/k
f1_average=(f1_av/n)/k
roc_auc_average=(roc_auc_av/n)/k
cm_average=(cm_av/n)



#print("acc_average=",acc_average)
#print("f1_average=",f1_average)
#print("roc_auc_av=",roc_auc_average)
#print("time_average=",time_average)
#print("train_score_av=",train_score_av)
#print("test_score_av=",test_score_av)



test_score= SVM.score(X_ts, y_ts)
print('SVM Test Score is : ' , test_score )
y_pred2 = SVM.predict(X_ts)

###### metrics to evaluate test set 
accuracy_test=metrics.accuracy_score(y_ts, y_pred2)
f1_sco_test=f1_score(y_ts, y_pred2)
roc_auc_test = metrics.roc_auc_score(y_ts, y_pred2)
cm_test = confusion_matrix(y_ts, y_pred2) 
Precision_test = precision_score(y_ts, y_pred2)       
Recall_test=recall_score(y_ts, y_pred2)
spec_test = cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])    
kappa_test=cohen_kappa_score(y_ts, y_pred2)
 
print("precision:",Precision_test)        
print("recall:",Recall_test) 
print("spec:",spec_test)                          
print("accuracy:",accuracy_test)
print("f1 score:",f1_sco_test)
print("auc:",roc_auc_test)
print( cm_test)


import seaborn as sns
sns.heatmap(cm_test,center=True, annot=True)

import time
from sklearn import metrics
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
LR= LogisticRegression()
t0 = time.time()
LR.fit(X_train1, y_train1)
fit_time = time.time() - t0
train_score= LR.score(X_train1, y_train1)
test_score=LR.score(X_test1, y_test1)
y_pred = LR.predict(X_test1)
cm1 = confusion_matrix(y_test1, y_pred) 
print('Confusion Matrix : \n', cm1)
cm_av+=cm1
acc_arr_S8=[]        
accuracy_score=metrics.accuracy_score(y_test1, y_pred)
acc_arr_S8.append(accuracy_score)
print("accuracy_score=",accuracy_score)
acc_av+=accuracy_score
from sklearn.metrics import f1_score
f1_sco=f1_score(y_test1, y_pred)
print('f1_score=',f1_sco)        
f1_av+=f1_sco
roc_auc = metrics.roc_auc_score(y_test1,y_pred)
print("roc_auc=",roc_auc)
roc_auc_av+=roc_auc
PrecisionScore = precision_score(y_test1, y_pred)
print("Precision:",PrecisionScore)
precision_av+=PrecisionScore
Rec=recall_score(y_test1, y_pred)
print("recall:", Rec)
Recall_av+=Rec


import seaborn as sns
sns.heatmap(cm1,center=True, annot=True)

import matplotlib.pyplot as plt
Y = [0.7726677148846961,0.8821070234113713]
x= ['SVM','LR']
plt.bar(x,Y,color="maroon",width=0.5)
plt.xlabel('ML algorithms')
plt.ylabel("Values")
plt.title('Comparing auc')
plt.show()


import matplotlib.pyplot as plt
Y = [0.784,0.8775510204081632]
x= ['SVM','LR']
plt.bar(x,Y,color="blue",width=0.5)
plt.xlabel('ML algorithms')
plt.ylabel("Values")
plt.title('Comparing accuracy')
plt.show()


import matplotlib.pyplot as plt
Y = [0.7922077922077922,0.8148148148148148]
x= ['SVM','LR']
plt.bar(x,Y,color="green",width=0.5)
plt.xlabel('ML algorithms')
plt.ylabel("Values")
plt.title('Comparing precision')
plt.show()
