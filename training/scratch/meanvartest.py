from sklearn.externals.joblib import dump, load
import os




trs = load('ScalerParameters_train.joblib')
tss = load('ScalerParameters_test.joblib')
vds = load('ScalerParameters_validation.joblib')

for i in range(596):
    print("i="+str(i)+"; Mean: train,test,val ", trs.mean_[i], tss.mean_[i], vds.mean_[i])
    print("i="+str(i)+"; Vari: train,test,val ", trs.var_[i], tss.var_[i], vds.var_[i])
