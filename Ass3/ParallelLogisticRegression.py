# -*- coding: utf-8 -*-
import numpy as np
import argparse
from time import time
from SparseVector import SparseVector
from LogisticRegression import readBeta,writeBeta,gradLogisticLoss,logisticLoss,lineSearch
from operator import add
from pyspark import SparkContext
import pickle
import os

def readDataRDD(input_file,spark_context):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)  

         x is a dictionary of the form:                 

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The return value is an RDD containing tuples of the form
                 (SparseVector(x),y)             

    """ 
    return spark_context.textFile(input_file)\
                        .map(eval)\
                        .map(lambda (x,y):(SparseVector(x),y))



	
def getAllFeaturesRDD(dataRDD):                
    """ Get all the features present in grouped dataset dataRDD.
 
	The input is:
            - dataRDD containing pairs of the form (SparseVector(x),y).  

        The return value is an list containing the union of all unique features present in sparse vectors inside dataRDD.
    """
    all_features = dataRDD.map(lambda (x,y): x)\
                          .reduce(add)

    return all_features.keys()

def totalLossRDD(dataRDD,beta,lam = 0.0):
    """  Given a sparse vector beta and a dataset in the form of RDD compute the regularized total logistic loss :
              
               L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2             
        
         Inputs are:
            - dataRDD: an RDD containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ

        The return value is a float number represents the total loss
    """
    loss = dataRDD.map(lambda (x,y): logisticLoss(beta, x, y))\
                  .reduce(add)
    return loss + lam * beta.dot(beta)

def gradTotalLossRDD(dataRDD,beta,lam = 0.0):
    """  Given a sparse vector beta and a dataset in the form of RDD, compute the gradient of regularized total logistic loss :
            
              ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β   
        
         Inputs are:
            - data: an RDD containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ

        The return value is a float number represents the gradient of total loss
    """ 
    gradLoss = dataRDD.map(lambda (x, y): gradLogisticLoss(beta, x, y))\
                      .reduce(add)
    return gradLoss + 2.0*lam*beta



def test(dataRDD,beta):
    """ Output the quantities necessary to compute the accuracy, precision, and recall of the prediction of labels in a dataset under a given β.
        
        The accuracy (ACC), precision (PRE), and recall (REC) are defined in terms of the following sets:

                 P = datapoints (x,y) in data for which <β,x> > 0
                 N = datapoints (x,y) in data for which <β,x> <= 0
                 
                 TP = datapoints in (x,y) in P for which y=+1  
                 FP = datapoints in (x,y) in P for which y=-1  
                 TN = datapoints in (x,y) in N for which y=-1
                 FN = datapoints in (x,y) in N for which y=+1

        For #XXX the number of elements in set XXX, the accuracy, precision, and recall of parameter vector β over data are defined as:
         
                 ACC(β,data) = ( #TP+#TN ) / (#P + #N)
                 PRE(β,data) = #TP / (#TP + #FP)
                 REC(β,data) = #TP/ (#TP + #FN)

        Inputs are:
             - dataRDD: an RDD containing pairs of the form (x,y)
             - beta: vector β

        The return values are
             - ACC, PRE, REC
       
    """
    P_set = dataRDD.filter(lambda (x,y): x.dot(beta) > 0).cache()
    N_set = dataRDD.filter(lambda (x,y): x.dot(beta) <= 0).cache()
    P = P_set.count()
    N = N_set.count()
    TP = P_set.filter(lambda (x,y): y == 1).count()
    FP = P - TP
    TN = N_set.filter(lambda (x,y): y == -1).count()
    FN = N - TN
    ACC = (TP + TN) / float(P + N)
    PRE = TP / float(TP + FP)
    REC = TP / float(TP + FN)
    return ACC, PRE, REC

def save_file_pickle(name, l):
    with open(name, 'wb') as f:
        pickle.dump(l, f)

def train(dataRDD,beta_0,lam,max_iter,eps,test_data=None,save_dir=''):
    """ Perform parallel logistic regression:

        to  minimize
  
             L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2  

    where the inputs are:
             - dataRDD: an rdd containing pairs of the form (x,y) as train data
             - beta_0: the starting vector β
             - lam:  is the regularization parameter λ
             - max_iter: maximum number of iterations of gradient descent
             - eps: upper bound on the l2 norm of the gradient
             - test_data: an rdd containing pairs of the form (x,y) as test data 

    The function returns:
         -beta: the trained β, 
         -gradNorm: the norm of the gradient at the trained β, and
         -k: the number of iterations performed
    """ 
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    beta_list = []
    gradNorm_list = np.zeros((max_iter, 1))
    acc_list = np.zeros((max_iter, 1))
    pre_list = np.zeros((max_iter, 1))
    rec_list = np.zeros((max_iter, 1))
    time_list = np.zeros((max_iter, 1))
    while k<max_iter and gradNorm > eps:
        obj = totalLossRDD(dataRDD,beta,lam)   

        grad = gradTotalLossRDD(dataRDD,beta,lam)  
        gradNormSq = grad.dot(grad)
        gradNorm = np.sqrt(gradNormSq)

        fun = lambda x: totalLossRDD(dataRDD,x,lam)
        gamma = lineSearch(fun,beta,grad,obj,gradNormSq)
        
        beta = beta - gamma * grad
        if test_data == None:
            print 'k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,'\tgamma = ',gamma
        else:
            acc,pre,rec = test(test_data,beta)
            print 'k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,'\tgamma = ',gamma,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec
        
        gradNorm_list[k] = gradNorm
        acc_list[k] = acc
        pre_list[k] = pre
        rec_list[k] = rec
        time_list[k] = time()-start
        beta_list.append(beta)
        k = k + 1
    # save the lists
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file_pickle(os.path.join(save_dir, "time.pkl"), time_list)
    save_file_pickle(os.path.join(save_dir, "gradNorm.pkl"), gradNorm_list)
    save_file_pickle(os.path.join(save_dir, "acc.pkl"), acc_list)
    save_file_pickle(os.path.join(save_dir, "pre.pkl"), pre_list)
    save_file_pickle(os.path.join(save_dir, "rec.pkl"), rec_list)
    save_file_pickle(os.path.join(save_dir, "beta.pkl"), beta_list)

    return beta,gradNorm,k

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Logistic Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('traindata',default=None, help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.1, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.') 
    parser.add_argument('--save_dir', type=str, default='', help='Where to save the statistics') 

    args = parser.parse_args()
    
    sc = SparkContext(appName='Parallel Logistic Regression')

    print 'Reading training data from',args.traindata
    traindata = readDataRDD(args.traindata, sc).cache()
    print 'Read',traindata.count(),'data points with',len(getAllFeaturesRDD(traindata)),'features in total'
    
    if args.testdata is not None:
        print 'Reading test data from',args.testdata
        testdata = readDataRDD(args.testdata, sc).cache()
        print 'Read',testdata.count(),'data points with',len(getAllFeaturesRDD(testdata)),'features'
    else:
        testdata = None

    beta0 = SparseVector({})

    print 'Training on data from',args.traindata,'with λ =',args.lam,', ε =',args.eps,', max iter = ',args.max_iter
    beta, gradNorm, k = train(traindata,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps,test_data=testdata,save_dir=args.save_dir) 
    print 'Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps
    print 'Saving trained β in',args.beta
    writeBeta(args.beta,beta)
