#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:23:31 2021
@author: groth
"""

import numpy as np

# initial probability vector
# probability of beginning in each (hidden) state
pi = np.array([0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.051, 0.2026])

# transition probability matrix
# a_ij probability of transitioning from state i to state j
A = np.array([[0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.009, 0.0025],
              [0.0008, 0.0020, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041],
              [0.0322, 0.0005, 0.0005, 0.0837, 0.0615, 0.0514, 0.2231],
              [0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036],
              [0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068],
              [0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479],
              [0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]])

# emission probability matrix
# b_j(o_t) # probability of emitting an observation at time t from state j
B = np.array([[0.000032, 0, 0, 0.000048, 0],
              [0, 0.308431, 0, 0, 0],
              [0, 0.000028, 0.000672, 0, 0.000028],
              [0, 0, 0.000340, 0, 0],
              [0, 0.0002, 0.0002223, 0, 0.002337],
              [0, 0, 0.010446, 0, 0],
              [0, 0, 0, 0.506099, 0]])

def viterbi(A, B, pi):
    ''' 
    decoding algorithm - calculates the most probable path (hidden states)
    
    input:
    transition probability matrix A (rows: states, columns: states): np array
    emission probability matrix B (rows: states, columns: obs): np array
    initial probability vector pi 
    
    returns:
    dict:
        BESTPATH: list
        BESTHPATHPROB: float
        VITERBI_MATRIX: np array (float)
        BACKPOINTER: np array (int)
    '''
    N = B.shape[0]
    T = B.shape[1]
    backpointer = np.zeros((N, T))
    viterbi = np.zeros((N, T))
    for i in range(N-1): #initialization
        viterbi[i, 0] = pi[i] * B[i, 0]
        backpointer[i, 0] = 0
    '''
    for each time step and each state, 
    calculate the viterbi probability and the backpointer index
    
    using the values computed from the previous column, 
    multiply each value by its transition and emission probability
    and take the maximum as the next cell's value
    
    the backpointer is the argmax,
    the state in the previous column that most likely leads to the current one
    
    viterbi[s,t]: max (over all states s') viterbi[s',t-1]* a[i,j] * b[s,t]
    backpointer[s,t] = argmax (over all states) viterbi[s',t-1]* a[i,j] * b[s,t]
    '''
    for t in range(1, T): 
        for s in range(N):
            step = [viterbi[j,t-1] * A[j,s] * B[s,t] for j in range(T)]
            viterbi[s, t] = np.max(step) 
            backpointer[s, t] = np.argmax(step)
    '''
    the best path starts from the last state; take the argmax of that column
    '''
    bestpathprob = np.max(viterbi[:, T-1])
    bestpathpointer = np.argmax(viterbi[:, T-1])
    
    '''
    use recursion to find the most probable path using the backpointers
    '''
    def backtrace(i,j,trace=None):
        if j == 0: 
            return
        else:
            goto = int(backpointer[i,j])
            if trace is None:
                trace = [goto]
            else:
                trace.append(goto)
            backtrace(goto, j-1, trace)
        return trace

    bestpath = [bestpathpointer] + backtrace(i = bestpathpointer, j = T-1)
    return {"BESTPATH": list(reversed(bestpath)), 
            "BESTPATHPROB": bestpathprob,
            "VITERBI_MATRIX": viterbi,
            "BACKPOINTER": backpointer}

result = viterbi(A, B, pi)
print(np.array_str(result['VITERBI_MATRIX'], precision = 2))
print(np.array_str(result['BACKPOINTER'], precision = 2))
print(result['BESTPATH'])