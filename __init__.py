import pandas as pd
import numpy as np
from Logisticlearning import Logisticlearning
import matplotlib.pyplot as plt
from pandas import scatter_matrix
from  pandas import plotting 
import sklearn.svm as svm
import importlib
Axes3D = importlib.import_module('mpl_toolkits.mplot3d').Axes3D


def testCostFunction():
    theta_t =np.mat([-2,-1,1,2]).T;
    a = np.ones((5,1))
    b= (np.arange(15).reshape(3,5).T+1)/10.0
    X_t = np.concatenate((a,b),axis=1)
    X_t = np.mat(X_t);
    y_t = np.mat([1,0,1,0,1]).T
    actor = Logisticlearning()
    '''
    lam = 3
    J = actor.lrCostFunction(theta_t, X_t, y_t, lam)
    grad = actor.gradFunction(theta_t, X_t, y_t, lam)
    '''
    lam = 0.1
    all_theta = actor.oneVsAll(X_t, y_t, 2, lam)
    pred = actor.predicOneVsAll(all_theta, X_t)
    y = np.mat(y_t)
    print pred
    print  ((y==pred)*1).mean();
def plot():
    data = pd.read_csv('train_data.csv')
    #print dir(data)
    data = np.array(data)
    #rownum = data.shape[0]
    columnNum = data.shape[1]
    X = data[3000:5000,19:41]
    #print(X.shape)
    y = np.array(data[3000:5000,-1])
    actor = Logisticlearning()
    X = actor.normalizeFeature(X)
    U,S = actor.PCD(X)
    K = 2
    Z = actor.projectData(X,U,K)
    
    pos = np.where(y==2)
    neg = np.where(y!=2)
    #print(pos)
    print(Z.shape)
    #print(Z[pos,:].shape)
    if K <3 :
        p1 = plt.plot(Z[pos,0],Z[pos,1],marker='o',markersize=9,color ='k')[0]
        p2 = plt.plot(Z[neg,0],Z[neg,1],marker='+',markersize =7,color='y')[0]
        plt.xlabel('type 1')
        plt.ylabel('type!=1')
    else:
        plt.contour(Z[:,0],Z[:,1],Z[:,2]).collections[0]
        #fig = plt.figure()
        
        #ax = Axes3D(fig)
        #ax.plot_surface(Z[:0],Z[:1],Z[:2])
    #print(y.shape)
    #plotting.scatter_matrix(X)
    #fig.colorbar(cax)
    plt.show()
def run():
    data = pd.read_csv('train_data.csv')
    data = np.mat(data)
    rownum = data.shape[0]
    columnNum = data.shape[1]
    X = data[0:12000,19:columnNum-1]
    y = data[0:12000,-1]
    #y = y.flatten();
    y_val = data[12000:12200,-1]
    X_val = data[12000:12200,19:columnNum-1]
    lam = 0.3
    num_labels = 3
    X = np.mat(X)
    m = X.shape[0]
    #normalize polyfeatures
    actor = Logisticlearning()
    #polydata = actor.mapFeature(X)
    #X_norm = actor.normalizeFeature(polydata)
    X_norm = X
    print m
    print X_norm.shape
    X = np.concatenate((np.ones((m,1)),X_norm),axis=1)
    
    all_theta=actor.oneVsAll(X,y,num_labels, lam)
    pred = actor.predicOneVsAll(all_theta, X)
    y = np.mat(y)
    print  ((y==pred)*1.0).mean();
    m_val = X_val.shape[0]
    X_val = np.concatenate((np.ones((m_val,1)),X_val),axis=1)
    predCross = actor.predicOneVsAll(all_theta, X_val)
    print ((y_val==predCross)*1.0).mean();
    '''actor = Logisticlearning()
    actor.processData()
    '''
def run2():
    data = pd.read_csv('train_data.csv')
    data = np.mat(data)
    rownum = data.shape[0]
    print rownum
    num_labels =3
    columnNum = data.shape[1]
    X = data[12000:24000,19:columnNum-1]
    y = data[12000:24000,-1]
    
    #y=y.flatten()
    actor = Logisticlearning()
    X = actor.mapFeature(X)
    X = actor.normalizeFeature(X)
    
    lam = 0.01
    
    all_theta = actor.OneVsAll2(X, y, num_labels, lam)
    pred = actor.predicOneVsAll2(all_theta, X)
    print('Training Set Accuracy: {:f}'.format((np.mean(pred == y%10)*100)))
    theta_csv = pd.DataFrame(all_theta)
    theta_csv.to_csv('theta_csv',index=False)
    sub = pd.read_csv('test_data.csv')
    sub = np.mat(sub)
    n = sub.shape[1]
    X_test = sub[:,19:n]
    X_test = actor.normalizeFeature(X_test)
    '''
    y_val = data[1000:2000,-1]
    X_val = data[1000:2000,1:columnNum-1]
    m_val = X_val.shape[0]
    X_val = np.concatenate((np.ones((m_val,1)),X_val),axis=1)
    predCross = actor.predicOneVsAll(all_theta, X_val)
    '''
    '''
    m_test = X_test.shape[0]
    X_test = actor.mapFeature(X_test)
    X_test = actor.normalizeFeature(X_test)
    X_test = np.concatenate((np.ones((m_test,1)),X_test),axis=1)
    predTest = actor.predicOneVsAll(all_theta, X_test)
    res = res = pd.DataFrame([])
    sub = pd.DataFrame(sub)
    res['connection_id']=sub.ix[:,0]
    res['target'] = pd.DataFrame(predTest).ix[:,0]
    '''
    #res.to_csv('sub4.csv', index=False)
    #print ((y_val==predCross)*1.0).mean();
def showData():
    sub = pd.read_csv('test_data.csv')
    sub = np.mat(sub)
    print sub[0:5,:]
    m = sub.shape[0]
    res = pd.DataFrame([])
    sub = pd.DataFrame(sub)
    #print type(sub.ix[:,0])
    res['connection_id']=sub.ix[:,0]
    res['target'] = sub.ix[:,1]
    
    res.to_csv('sub1.csv',index=False)
def svmMethod():
    data = pd.read_csv('train_data.csv')
    data = np.array(data)
    rownum = data.shape[0]
    print rownum
    
    columnNum = data.shape[1]
    X = data[12000:24000,19:columnNum-1]
    y = data[12000:24000,-1]
    actor = Logisticlearning()
    X = actor.normalizeFeature(X)
    #X = X.astype(str)
    y = y.astype(str)
    gamma = 0.001
    C = 10000
    clf = svm.SVC(C,kernel="rbf",verbose=2)
    
    clf.fit(X,y)
    pred = clf.predict(X)
    print('Training Set Accuracy: {:f}'.format((np.mean(pred == y)*100)))
    
    sub = pd.read_csv('test_data.csv')
    sub = np.mat(sub)
    n = sub.shape[1]
    X_test = sub[:,19:n]
    X_test = actor.normalizeFeature(X_test)

    
    m_test = X_test.shape[0]
    #X_test = actor.mapFeature(X_test)
    X_test = actor.normalizeFeature(X_test)
    #X_test = np.concatenate((np.ones((m_test,1)),X_test),axis=1)
    #predTest = actor.predicOneVsAll(all_theta, X_test)
    predTest = clf.predict(X_test)
    res = res = pd.DataFrame([])
    sub = pd.DataFrame(sub)
    res['connection_id']=sub.ix[:,0]
    res['target'] = pd.DataFrame(predTest).ix[:,0]
    
    res.to_csv('sub4.csv', index=False)
    #print ((y_val==predCross)*1.0).mean();

#svmMethod()
#run2() ;
plot()
#testCostFunction()
#showData()