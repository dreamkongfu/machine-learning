
import numpy as np
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
from numpy import int16
from scipy import optimize
from scipy.optimize import minimize
from scipy.special import expit 
import scipy.linalg as linalg

class Logisticlearning(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def processData(self):
            data = pd.read_csv('train_data.csv')
            print data.head()
    
                
    def sigmoid(self,z):
        z = np.mat(z,dtype=np.float128)
        #z[z==0]=1
        g = 1.0 / (1.0+ np.exp(-z))
        return g
    def sigmoid2(self,z):
        g= np.zeros(z.shape)
        g = expit(z)
        return g
    def sigmoid3(self,z):
        g= np.zeros(z.shape)
        g = 1/(1 + np.exp(-z))
        return g
    def mapFeature(self,X):
        X = np.mat(X)   
        degree = 2  
        n = X.shape[1]
        m = X.shape[0]
        out = np.mat(np.ones((m,n*(n-1)*5/2)))
        print n 
        index = 0 
        for i in range(1,degree+1):
            for j in range(0,i+1):
                for k in range(0,n):
                    for h in range(k+1,n):
                        X1=X[:,k]
                        X2=X[:,h]                    
                        newFeature = np.multiply(np.power(X1,j),np.power(X2,i-j));
                        out[:,index] = newFeature
                        index = index + 1
                        print h
        
        print index                 
        return out 
    def normalizeFeature(self,polydata):
        print "===begin normalizeFeature"
        polydata = np.mat(polydata,dtype=np.float128)
        mu = np.mean(polydata)
        #mu = np.einsum('ij->i',polydata)[:,None]/polydata.shape[1]
        
        std = np.std(polydata)*1.0
        demean = np.subtract(polydata,mu)
        normPoly = np.divide(demean,std)
        print "===end normalizeFeature"
        return normPoly     
    def lrCostFunction(self,theta,X,y,lam):
        #print theta.shape
        theta = np.mat(theta).T
        m = y.shape[0]
        J = 0;        
        
        I = np.mat(np.ones((m,1)))     
        thetaOther = theta[1:]
        h = self.sigmoid( np.dot(X,theta))
        thetaOther2 = np.multiply(thetaOther,thetaOther)
        h=np.mat(h,dtype=np.float128)
        h[h==0]=1
        logh = np.log(h)
        Ih = I-h
        Ih[Ih==0]=1
        Ih = np.mat(Ih,dtype=np.float128)
        logIh = np.log(Ih)   
        J1 = lam*1.0/(2*m)*np.sum(thetaOther2)
       
        J2 = (np.dot(-y.T,logh) - np.dot((I-y).T,logIh))*1.0/m
        print J1
        print J2
        J = J+J1+J2.max()
        theta = np.array(theta)
        theta = theta[:,0]
        #J = J + 1/m*(-y.T*(logh-(I-y)).T*logIh)+lam/(2*m)*sum(thetaOther2);
        '''grad0= 1.0/m*(X[:,0].T*(h-y))
        grad1=1.0/m*(X[:,1:].T*(h-y))+lam*1.0/m*thetaOther;
        grad = np.concatenate((grad0,grad1),axis=0)
        '''
        print J
        return J
    def lrCost(self,theta,X,y,lam,return_grad=False):
        m = len(y)
        J = 0
        grad = np.zeros(theta.shape)
        #=====
        test = np.transpose(np.log(self.sigmoid(np.dot(X,theta))))
        Ih = 1-self.sigmoid(np.dot(X,theta)) 
        one = (y.T)*np.transpose(np.log(self.sigmoid(np.dot(X,theta))))
        two = np.transpose(1-y)*np.transpose(np.log(1-self.sigmoid(np.dot(X,theta))))
        reg =(float(lam)/(2*m))*np.power(theta[1:theta.shape[0]],2).sum()
        J=-(1./m)*(one+two).sum()+reg
        print J
        #grad = (1./m)*np.dot(X.T,self.sigmoid(np.dot(X,theta))-y).T+ ( float(lam) / m )*theta
        #grad_no_regularization = (1./m) * np.dot(X.T,self.sigmoid( np.dot(X,theta) ) - y)
        #grad[0] = grad_no_regularization[0]
        grad = self.gradFunction(theta, X, y, lam)
        print J
        if return_grad:
            return J, grad
        else:
            return J
    def gradFunction(self,theta,X,y,lam):
        theta = np.mat(theta).T
        m = y.shape[0]
        thetaOther = theta[1:]
        h = self.sigmoid(np.dot(X,theta))
        grad = np.mat(np.zeros(theta.shape,int16))
        test = np.dot(X[:,0].T,(h-y));
        
        grad0= (np.dot(X[:,0].T,(h-y)))*1.0/m
        grad1=(np.dot(X[:,1:].T,(h-y)))*1.0/m+thetaOther*lam*1.0/m
        grad = grad+np.concatenate((grad0,grad1),axis=0)
        
        grad = np.array(grad)
        grad = grad[:,0]
        #print grad
        theta = np.array(theta.T)
        theta = theta[:,0]
        #print theta
        return grad
    
    def PCD(self,X):
        m,n = X.shape
        U = np.zeros(n)
        S = np.zeros(n)
        sigma = (1.0/m)*(X.T).dot(X)
        U,S,Vh = linalg.svd(sigma)
        S = linalg.diagsvd(S,len(S),len(S)) # 
        return U,S
    def projectData(self,X,U,K):
        Z = np.zeros((X.shape[0],K))
        U_reduce = U[:,:K]
        Z = X.dot(U_reduce)
        return Z
    def oneVsAll(self,X,y,num_labels,lam):
        print "===begin oneVsAll"
        m = X.shape[0]
        n= X.shape[1]
        
        all_theta = np.zeros((num_labels,n))
        #
        #X = np.mat(X,int16)
        y = np.mat(y)
        for c in range(0,num_labels):
            #initial_theta = np.zeros((n+1,1),int16)
            initial_theta = np.zeros((n,1))
            #print initial_theta.shape
            #print initial_theta
            print "==="
            x0 = initial_theta
            args=(X,(y==c)*1,lam)
            thetai = optimize.fmin_cg(self.lrCostFunction,x0,fprime=self.gradFunction,args=args,maxiter=300)
            #print (thetai.T).shape
            all_theta[c,:]= thetai
            print c
                
        #all_theta = all_theta.reshape(num_labels,n+1)
        #all_theta = all_theta.reshape(num_labels,n)
        print all_theta
        return all_theta
    def OneVsAll2(self,X,y,num_labels,lam):
        m,n = X.shape
        
        all_theta = np.zeros((num_labels,n+1))
        X= np.column_stack((np.ones((m,1)),X))
        for c in xrange(num_labels):
            initial_theta = np.zeros((n+1,1))
            myargs =(X,(y==c)*1,lam,True)
            theta = minimize(self.lrCost,x0=initial_theta,args=myargs,options={'disp':True,'maxiter':100},method='Newton-CG',jac=True)
            #theta = np.array(theta)
            print theta
            all_theta[c,:]=theta['x']
        return all_theta
    def predicOneVsAll(self,all_theta,X):
        print "====predictBegin"
        m = X.shape[0]
        num_labels= all_theta.shape[0]
        all_theta = np.mat(all_theta)
        #X = np.concatenate((np.ones((m,1)),X),axis = 1)
        prob = self.sigmoid(X*(all_theta.T))
        print prob[0:3,:]
        res = prob.argmax(1)
        print res
        return res
    def predicOneVsAll2(self,all_theta,X):
        m = X.shape[0]
        num_labels = all_theta.shape[0]
        p=np.zeros((m,1))
        X = np.column_stack((np.ones((m,1)),X))
        p= np.argmax(self.sigmoid(np.dot(X,all_theta.T)), axis=1)  
        return p      