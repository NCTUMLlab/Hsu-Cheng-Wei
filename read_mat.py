# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 18:23:34 2016

@author: hcwei
"""


#import scipy.io
#dev = scipy.io.loadmat('dev.mat')
#for key,value in dev.iteritems():
#    print key

from __future__ import absolute_import
from __future__ import division
import scipy.io
import numpy as np
import collections
#import tensorflow as tf

class DataSet(object):

  def __init__(self, ivectors, ids , durations , verify_data=None , verify = False ,target_status=None, trial_set=None, one_hot=False,
               dtype=np.float32):#dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    #dtype = tf.as_dtype(dtype).base_dtype
    #if dtype not in (tf.float32):
    #  raise TypeError('Invalid ivector dtype %r, expected float32' %
    #                  dtype)
       

    if verify == False :
      assert ivectors.shape[0] == ids.shape[0], ('ivectors.shape: %s labels.shape: %s' % (ivectors.shape,labels.shape))
      self._num_examples = ivectors.shape[0]

      #change ids to label

      labels_set = np.unique(ids)
      labels_list_set = list(labels_set)

      ids_list = ids.tolist()
      labels = [labels_list_set.index(x) for x in ids_list]
      #labels = [labels_list_set.index(x) for x in ids]
    
    else : 
      labels = None 

    #setting data

    
    ivectors = ivectors.astype(np.float32)
    
    self._ori_ivectors = ivectors
    self._ivectors = ivectors
    
    self._ori_ids = ids
    #self._ids = ids
    
    self._ori_labels = labels
    self._labels = labels
    
    self._ori_durations = durations
    
    self._verify_data = verify_data    
    self._verify = verify
    
    
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
    self._target_status = target_status
    self._trial_set=trial_set
   
    self._mean_ivec = None

    self._n_speaker = 0
   
    self._W = None

  @property
  def ori_ivectors(self):
    return self._ori_ivectors

  @property
  def ori_labels(self):
    return self._ori_labels
    
  @property
  def ivectors(self):
    return self._ivectors

  @property
  def labels(self):
    return self._labels

  @property
  def mean_ivec(self):
    return self._mean_ivec

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed
    
  @property
  def ivector_dict(self):
    return self._ivector_dict
    
  @property
  def n_speaker(self):
    return self._n_speaker 

  @property
  def W(self):
    return self._W

  def eliminate(self,time):
    """Eliminate ivectors smaller than times"""
    idx = [i for i,v in enumerate(self._ori_durations) if v > time]
    self._ivectors = [self._ori_ivectors[i] for i in idx]
    self._labels = [self._ori_labels[i] for i in idx]
    #self._ids = [self._ori_ids[i] for i in idx]
    
      
  def speaker_dict(self):
    """construct speaker dictionary data with same speaker """
    ivector_dict =collections.defaultdict(list)
    for (i,l) in zip(self._ivectors, self._labels):
        ivector_dict[l].append(i)
    self._n_speaker = len(ivector_dict)    
    #self._ivector_dict = ivector_dict
    self._ivector_dict = [np.vstack(ivector_dict[k]) for k in range(self._n_speaker)  ]
  
  def sort(self):
    """Sort speaker ivector dependent on labels"""
    [self._labels,self._ivectors] = [list(x) for x in zip(*sorted(zip(self._labels,self._ivectors ),key=lambda x: x[0]) )]
      
  def next_batch(self, batch_size,shuffle= True):
    """Return the next `batch_size` examples from this data set."""
    
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      
      # Shuffle the data
      if shuffle:
          perm = np.arange(self._n_speaker)
          np.random.shuffle(perm)
          self._ivectors = np.vstack([ self._ivector_dict[k] for k in perm])
          self._labels = [k  for k in perm for l in range(len(self._ivector_dict[k] ))] 
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._ivectors[start:end], self._labels[start:end]

  def length_norm(self):
    
    self._mean_ivec = np.mean(self._ivectors,axis = 0)
    minus_ivec = np.subtract(self._ivectors ,self._mean_ivec )
    ivec_norm = np.linalg.norm(minus_ivec,ord=2,axis=1)
    #print np.shape(ivec_norm)
    #data = minus_ivec / ivec_norm

    data = (minus_ivec.T  / ivec_norm).T

    cov_data = np.cov(data.T)
    U, D, V = np.linalg.svd(cov_data)
    W = V.T * np.diag( 1./(np.sqrt(np.diag(D))+1e-10  ) )
    self._ivectors=  np.dot(data , W)

    self._W  = W

    return W

  def test_length_norm(self,W,mean_ivec):

    minus_ivec = np.subtract(self._ivectors, mean_ivec )
    ivec_norm = np.linalg.norm(minus_ivec,ord=2,axis=1)

    minus_ver_ivec = np.subtract(self._verify_data, mean_ivec)
    ivec_ver_norm = np.linalg.norm(minus_ver_ivec,ord=2,axis=1)
    #print np.shape(ivec_norm)
    #data = minus_ivec / ivec_norm

    data = (minus_ivec.T  / ivec_norm).T
    
    ver_data = (minus_ver_ivec.T / ivec_ver_norm).T

    self._ivectors=  np.dot(data , W)
    self._verify_data = np.dot(ver_data, W)


  def compute_eer(self,scores):
     scores = np.reshape(scores,[-1,1])
     assert scores.shape[0] == self._target_status.shape[0], (
          'scores.shape: %s target_status.shape: %s' % (scores.shape,
                                                 self._target_status))
                                                 
     #[scores_sort,target_status_sort] = [list(x) for x in zip(*sorted(zip(scores,self._target_status),key=lambda x: x[0]) )]

     #x= [j for i,j  in zip(*sorted(zip(scores,self._target_status),key=lambda x: x[0]))]
     #[_ , x] = zip(*[list(x) for x in sorted( zip(scores , self._target_status),key=lambda x:x[0])  ])
    
     x = self._target_status[np.argsort(scores.T)]
     
    
     #x_1 = np.where(x==np.ones(len(x)),1,0 )

     #x_0 = np.where(x==np.ones(len(x)),0,1 )
      
    

     #x_1 = [  i for i in x]

     #x_0 = [1-i for i in x]
     x_1 = 0+x
     x_0 = 1-x


     FN = np.true_divide(np.cumsum(x_1) , (np.sum(x_1)+1e-10))
     TN = np.true_divide(np.cumsum(x_0) , (np.sum(x_0)+1e-10))
     FP = np.subtract(1. , TN)
     TP = np.subtract(1. , FN)
     
     FNR = np.true_divide(FN,np.add(np.add(TP,FN),1e-10))
     FPR = np.true_divide(FP,np.add(np.add(TN,FP),1e-10))
     difs = np.subtract(FNR,FPR)
     #idx1 = [i for i,v in enumerate(difs) if v<0][-1]
     #idx2 = [i for i,v in enumerate(difs) if v>=0][0]

     idx1 = np.where( difs <0 )[0][-1]
     idx2 = np.where( difs>=0 )[0][0]
     
     #print idx1
     #print idx2

     x = [FNR[idx1],FPR[idx1]]
     y = [FNR[idx2],FPR[idx2]]
     a = np.true_divide(( x[0]- x[1] ) ,( y[1] - x[1] - y[0] + x[0]))
     eer = 100. * ( x[0] + a * ( y[0] - x[0] ) )
     return eer     
     
  def compute_dcf(self,scores):
    scores = np.reshape(scores,[-1,1])
    assert scores.shape[0] == self._target_status.shape[0], (
          'scores.shape: %s target_status.shape: %s' % (scores.shape,
                                                 self._target_status))
    
    assert scores.shape[0] == self._trial_set.shape[0], (
          'scores.shape: %s trial_set: %s' % (scores.shape,
                                                  self._trial_set.shape))
                                
    #[target_status_prog, scores_prog, _]= zip(*[ list(x) for x in  zip(self._target_status, scores, self._trial_set) if x[2]==0  ])
    #[target_status_eval, scores_eval, _]= zip(*[ list(x) for x in  zip(self._target_status, scores, self._trial_set) if x[2]==1  ])
   
    #list_eval = [ list(y) for y in  zip(self._target_status, scores, self._trial_set) if y[2]==1  ]
    #list_eval_zip = zip(self._target_status, scores, self._trial_set)
    #list_eval = [ list(y) for y in  list_eval_zip if y[2]==1]
    #list_eval_first = list_eval_zip[0:3000000]
    #list_eval_last = list_eval_zip[3000000:scores.shape[0]+1]
    
    #[target_status_eval_first,scores_eval_first,_] = zip(*list_eval_first)
    #[target_status_eval_last,scores_eval_last,_] = zip(*list_eval_last)
   
    #target_status_eval = target_status_eval_first + target_status_eval_last
    #scores_eval = scores_eval_first + scores_eval_last
    #[target_status_eval, scores_eval, _]= zip(*list_eval)
    

    ####
    #list_eval = [ list(y) for y in  zip(self._target_status, scores, self._trial_set) if y[2]==1  ]
    #[target_status_eval, scores_eval, _]= zip(*list_eval)

    #list_prog = [ list(x) for x in  zip(self._target_status, scores, self._trial_set) if x[2]==0  ]
    #[target_status_prog, scores_prog, _]= zip(*list_prog)

    ####
    #print(scores.shape[0])
    #print(self._target_status.shape[0])
    #print(self._trial_set.shape[0])

    #[target_status_prog, scores_prog, _]= zip(*[ list(x) for x in  zip(self._target_status, scores, self._trial_set) if x[2]==0  ])
   
 
    #[target_status_eval, scores_eval, _]= zip(*[ list(x) for x in  zip(self._target_status, scores, self._trial_set) if x[2]==1  ])
   

    target_status_prog = self._target_status[ self._trial_set == 0 ]
    target_status_eval = self._target_status[ self._trial_set == 1 ]
    scores_prog = scores[ self._trial_set == 0 ] 
    scores_eval = scores[ self._trial_set == 1 ]


 
    ####
    #target_status_eval = []
    #scores_eval = []
    #for x in zip(self._target_status, scores, self._trial_set):
    #    if x[2]==1:
    #        print(len(target_status_eval))
    #        #print(len(scores_eval))
    #        target_status_eval.append(x[0])
    #        scores_eval.append(x[1])
    ####


    #[target_status_eval_first, scores_eval_first, _]= zip(*[ list(x) for x in  zip(self._target_status[0:6000000], scores[0:6000000], self._trial_set[0:6000000]) if x[2]==1  ])
    #[target_status_eval_second, scores_eval_second, _]= zip(*[ list(x) for x in  zip(self._target_status[6000000:scores.shape[0]+1], scores[6000000:scores.shape[0]+1], self._trial_set[6000000:scores.shape[0]+1]) if x[2]==1  ])
    #target_status_eval = target_status_eval_first + target_status_eval_second
    
    
    # progess subet
    #Status = [i for i,j in zip(*sorted( zip(target_status_prog,scores_prog),key=lambda x:x[1]))    ]
     
    #ori
    #[Status, _] = zip(*[list(x) for x in sorted( zip(target_status_prog,scores_prog),key=lambda x:x[1])  ])

    Status = target_status_prog[scores_prog.argsort()]

    
    #Status_1 = np.where(Status==np.ones(len(Status)),1,0 )

    #Status_0 = np.where(Status==np.ones(len(Status)),0,1 )

    #Status_1 = [  i for i in Status]

    #Status_0 = [1-i for i in Status]
   
    Status_1 = Status
    Status_0 = 1-Status

    M = np.true_divide(np.cumsum(Status_1),np.sum(Status_1))

    F = np.subtract(1.,  np.true_divide(np.cumsum(Status_0),np.sum(Status_0)))

    #print(M)
    #print(F)
    
    dcf14_prog = np.amin( np.add(M,np.multiply(100.,F)  )  )
    
    #evaluation subset
    
    #Status = [i for i,j in zip(*sorted( zip(target_status_eval,scores_eval),key=lambda x:x[1]))    ]
    
    #ori
    #[Status, _] = zip(*[list(x) for x in sorted( zip(target_status_eval,scores_eval),key=lambda x:x[1])  ])
    
    Status = target_status_eval[scores_eval.argsort()]
    
    #Status_1 = np.where(Status==np.ones(len(Status)),1,0 )
    
    #Status_0 = np.where(Status==np.ones(len(Status)),0,1 )

    #Status_1 = [  i for i in Status]
    
    #Status_0 = [1-i for i in Status]

    Status_1 = Status
    Status_0 = 1-Status

    M = np.true_divide(np.cumsum(Status_1),np.sum(Status_1))

    F =  np.subtract(1.,np.true_divide(np.cumsum(Status_0),np.sum(Status_0)))

    #print(M)
    #print(F)
    
    dcf14_eval = np.amin( np.add(M,np.multiply(100,F)  )  )
    return dcf14_prog,dcf14_eval
     

def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=np.float32):#dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()  
  
  train_mat  = scipy.io.loadmat(train_dir+"dev.mat")
  train_struct = train_mat['dev'][0,0]
  train_ids = train_struct['labels']
  train_ivec = train_struct['ivectors']
  train_dur = train_struct['durations']
  
  verify_ivec_mat = scipy.io.loadmat(train_dir+'avg_model_ivec.mat')
  verify_ivec = verify_ivec_mat['avg_model_ivec']
  
  test_ivec_mat = scipy.io.loadmat(train_dir+'test.mat')
  test_struct = test_ivec_mat['test'][0,0]
  test_ids = None #test_struct['labels']
  test_ivec = test_struct['ivectors']
  test_dur = test_struct['durations']
  
  target_status_mat = scipy.io.loadmat(train_dir+"target_status.mat")
  target_status = target_status_mat['target_status']
  
  target_status  = np.reshape(target_status ,[-1,1])

  trial_set_mat = scipy.io.loadmat(train_dir+"trial_set.mat")
  trial_set = trial_set_mat['trial_set']
  trial_set = np.reshape(trial_set,[-1,1])


  data_sets.train = DataSet(train_ivec , train_ids, train_dur, dtype=dtype)
  data_sets.test = DataSet(test_ivec , test_ids , test_dur,verify_data = verify_ivec , verify = True, target_status=target_status, trial_set=trial_set, dtype=dtype)

  return data_sets
