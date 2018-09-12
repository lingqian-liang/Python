import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy import stats
import math
import datetime

"""ZL model use MC to simulate stock prices path which would be used as reference to check timing of converting bonds"""

def Input(T_,r_,sigma_,K_,path_,t_):
    global T,S0,r, sigma, N, K, dt,path,t
    T = T_  # total time
    r = r_
    sigma = sigma_
    t = t_# remaining time
    N = math.ceil(t*365)+20
    K = K_
    dt = 1/365.0
    path = path_

# Stock Path
def stock_path(stock_price):
    np.random.seed(path)
    S = np.zeros((N+1,path))
    time_pass = int((T-t)/dt)
    for n, element in enumerate(stock_price):
        S[n] = element
    for i in range(len(stock_price)-1,N):
        S[i+1] = S[i]*np.exp((r-sigma**2*0.5)*dt + sigma*np.sqrt(dt)* np.random.standard_normal(path))
    return S

# BS model to calculate call price
"""Owner of convertible bond has the rights to convert bond as stocks if the underlying price surplus the converting price. But people tend 
not to convert bonds until the end. Thus, we can take this right of converting as European call option"""
def BS_model_call(St,K,r,sigma,T_t):
    d1 = (np.log(St/K)+(r+sigma*sigma*0.5)*T_t)/(sigma*np.sqrt(T_t))
    d2 = d1 - sigma*np.sqrt(T_t)
    C = St*stats.norm.cdf(d1,0,1) - K*np.exp(-r*T_t)*stats.norm.cdf(d2,0,1)
    return C

# Compare each element of array
"""if array1 < array2, return True, else False"""
def compare_array(array1,array2):
    array1 = list(array1)
    array2 = list(array2)
    if len(array1)==len(array2):
        count = sum(a1<a2 for a1,a2 in zip(array1,array2))
        if count<len(array1):
            return False
        else: return True
    else: print('length doesn`t match')

# the time period before bonds being convertible
def time_convert(time1, time2):
    t1 = datetime.datetime.strptime(time1,'%Y-%m-%d')
    t2 = datetime.datetime.strptime(time2,'%Y-%m-%d')
    diff = t2-t1
    diff = diff.days
    tc = diff/365.0
    return tc

# Derive implied K (new convertible price)
"""using bisection to derive K"""
def computing_K(LEnd, REnd, Tgt, Acc, S,r,sigma,t):
    left=LEnd
    right=REnd
    mid = (left+right)/2.0
    y_left=BS_model_call(S,left,r,sigma,t)*100.0/left-Tgt
    y_mid=BS_model_call(S,mid,r,sigma,t)*100/mid-Tgt
    while mid-left>Acc:
        if (y_left * y_mid>0):
            left=mid
            y_left=y_mid
        else:
            right=mid
        mid=(left+right)/2.0
        y_mid = BS_model_call(S,mid,r,sigma,t)-Tgt
    return mid

# Compute convertible bond prices
"""In China, convertible bonds not only can sell back but also can call back. In order to avoid liquidity risk, issuer tend to lower converting
price K which would increase convertible bond value. In this function, I would consider three situation: changing K, issuer`s calling back and 
holding until end and then compute the present value of convertible bond. """
def convertible_bond_price(I,Bc,time1,time2,eps,T,S0,r,sigma,N,K,path,t,stock_price):
    S = np.zeros((N+1,path))
    S = stock_path(stock_price)
    V = np.zeros((N+1,path))
    K_new = K
    K_array = np.ones((N+1,path))*K_new
    # time1 is the issue timing of convertible bond, time2 is the timing of converting.
    tc = time_convert(time1,time2)
    if tc< T-t:
        # finding path nodes
        convert_ = 21
    else:
        convert_= int(math.ceil((tc+t-T)*365))
    t0 = 21
    # time to have interest
    t_i = np.arange(t,0,-1)
    t_i.sort()
    for j in range(0,path):
        i0 = 0
        for i in range(convert_,N+1,1):
            T_t = (N-i)*dt
            period = sum((i-21)/dt > inter for inter in t_i)
        #changing price
            # if stock prices lower than 0.7 times convertible prices in 20 consecutive days, the customer can sell back convertible bond
            if  i>=20+i0 and compare_array(S[i-20:i+1,j],0.7*K_array[i-20:i+1,j])and i<N:
                # customer sell back value = 100 + current interest
                Put = 100.0 +I[period]
                S_new = S[i,j]*100.0/K_array[i,j]
                C = BS_model_call(S[i,j],K_array[i,j],r,sigma,T_t)
                # convertible bond value = call + continuing holding value(maturity)
                V[i,j]=C*100.0/K_array[i,j] + (100.0 + sum(I[period:]))*np.exp(-r*(T_t))
                if V[i,j] < Put:
                    # if continuing holding value < sell back value, the issuer would change prices to aviod liquidity risk
                    C2 = (Put- (100.0 + sum(I[period:]))*np.exp(-r*(T_t)))
                    K_new = computing_K(0.1,K_array[i,j],C2,0.001,S[i,j],r,sigma,T_t)
                    # There is a boundary for issuer to lower its prices that is eps, each has its own boundary
                    if K_new >=lower_b:
                        K_array[i:,j]=K_new
                        i0=i
                    else:
                        # if the changing prices is lower than boundary, the issuer would give up changing prices
                        V[t0,j]=np.exp(-r*(i-t0)*dt)*Put
                        break
            # Normally, if the stock prices are higher than 1.3 times convertible prices in consecutive 15 days, the issuers have the right call back
            if i>=30 and compare_array(1.3*K_array[i-14:i+1,j],S[i-14:i+1,j]) and i<N:
                V[t0,j]=np.exp(-r*(i-t0)*dt)*(100*S[i,j]/K_array[i,j]+sum(I[0:period]))
                #print(V[t0,j])
                break
            if i == N:
                # at the end, the customer would choose sell back or convert to stocks
                V[i,j]=(max(100.0/K_array[i,j]*S[i,j],Bc)+sum(I[0:]))
                #print(V[i,j])
                V[t0,j]=np.exp(-r*t)*V[i,j]

    V0 = V[t0].mean()
    print(V0)
    return V0


# Data Input
Input(6.0,0.037889,0.296,7.43,1000,3.9699)
I =[1.0,1.5,1.6,2.0]
Bc = 106
time1='2016-01-18'
time2='2016-07-04'
lower_b = 3.37
T_=6.0
S0_=9.21
r_=0.037889
sigma_=0.296
N_=100
K_=7.43
path_=1000
t_=3.9699
# 20 days ahead of time stock prices because changing prices need to refer to 20-day previous prices.
stock_price=[8.63, 8.78, 8.55, 8.47, 8.52, 8.45, 8.52, 8.52, 8.22, 8.35, 8.32, 8.46, 9.08, 8.92, 9.08,
               9.07, 9.48, 9.47,9.44, 9.59, 9.21]
V0= convertible_bond_price(I,Bc,time1,time2,lower_b,T_,S0_,r_,sigma_,N_,K_,path_,t_,stock_price)