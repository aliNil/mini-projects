import numpy as np 


def ode_function(x, t, u):
    # polymerization reactor dynamic model
    F = 1
    dx = np.zeros(4,)
    dx[0] = 60 - 10*x[0] - 2.4568*x[0]*np.sqrt(x[1])
    dx[1] = 80*u - (0.1022+ 10*F)*x[1]
    dx[2] = 0.0024121*x[0]*np.sqrt(x[1]) + 0.112191*x[1] - 10*x[2]
    dx[3] = 245.978*x[0]*np.sqrt(x[1])- 10*x[3]
    return dx


def signal_generator(nstep,amp,freq):
  # Choose training data lenth

  # random signal generation

  # a_range = [0,2]
  a_range = amp
  a = np.random.random(nstep) * (a_range[1]-a_range[0]) + a_range[0] # range for amplitude
  a[0] = 0

  # b_range = [5, 20]
  b_range = freq
  b = np.random.random(nstep) *(b_range[1]-b_range[0]) + b_range[0] # range for frequency
  b = np.round(b)
  b = b.astype(int)

  b[0] = 0

  for i in range(1,np.size(b)):
      b[i] = b[i-1]+b[i]

  # Random Signal
  i=0
  random_signal = np.zeros(nstep)
  while b[i]<np.size(random_signal):
      k = b[i]
      random_signal[k:] = a[i]
      i=i+1
  return random_signal


def data_prep(u,y, window = 2, val_ratio = 0.7):
    Xn =  np.concatenate((u,y), axis=1)
    Yn = (y)
    
    cut_index = int(len(u)*val_ratio) # index number to separate the training and validation set
#     print(Xn)
#     
#     Yn_train = Y[0:cut_index]
#     Xn_val = X[cut_index:]
#     Yn_val = Y[cut_index:]
    
    
    X = []
    Y = []
    for i in range(window,len(u)):
        X.append(Xn[i-window:i,:])
        Y.append(Yn[i])
    
    X_train = X[:cut_index]
    Y_train = Y[:cut_index]
    X_val = X[cut_index:]
    Y_val = Y[cut_index:]
        
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_val, Y_val = np.array(X_val), np.array(Y_val)
    
    return X_train, X_val, Y_train, Y_val