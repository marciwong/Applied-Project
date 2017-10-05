#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:52:57 2017

@author: marcowong
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jun 25 21:34:14 2017

@author: marcowong
"""


import statsmodels.formula.api as smf
from statsmodels.stats import stattools
from statsmodels.stats import diagnostic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.tsa.stattools as ts
from scipy import stats

def coefficients(X,Y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    rSq = r_value**2
    coefficients = [intercept, slope, rSq, std_err]
    return coefficients

df = pd.read_csv('/Users/marcowong/Dropbox/Code/excessReturn.csv')
noofObs = len(df)
df = df.iloc[:noofObs,:]

noOfVar = len(df.columns)-1

columnNames = list(df.columns.values)

chowTest = []

#Chow test
for i in range(2,noOfVar+1):
    fullChow = []
    sub1Chow = []
    sub2Chow = []
    chowReg = pd.concat([df.iloc[:,1],df.iloc[:,i]],axis = 1)
    chowReg  = chowReg.dropna(how = 'any')
    secondPeriod = 30
    tempFullChow = coefficients(chowReg.iloc[:,1],chowReg.iloc[:,0])
    tempSub1Chow = coefficients(chowReg.iloc[0:secondPeriod-1,1],chowReg.iloc[0:secondPeriod-1,0])
    tempSub2Chow = coefficients(chowReg.iloc[secondPeriod:,1],chowReg.iloc[secondPeriod:,0])
    for j in range(0,len(chowReg)):
        fullChow.append(np.square(chowReg.iloc[j,0] - tempFullChow[0] - tempFullChow[1] * chowReg.iloc[j,1]))
    for j in range(0,secondPeriod):
        sub1Chow.append(np.square(chowReg.iloc[j,0] - tempSub1Chow[0] - tempSub1Chow[1] * chowReg.iloc[j,1]))
    for j in range(secondPeriod,len(chowReg)):
        sub2Chow.append(np.square(chowReg.iloc[j,0] - tempSub2Chow[0] - tempSub2Chow[1] * chowReg.iloc[j,1]))
    sumFullChow = np.sum(fullChow)
    sumSub1Chow = np.sum(sub1Chow)
    sumSub2Chow = np.sum(sub2Chow)
    chowTest.append(((sumFullChow-(sumSub1Chow+sumSub2Chow))/2)/((sumSub1Chow+sumSub2Chow)/(len(chowReg)-4)))



errorOLSSq = []
errorHMSq = []
dESS = []
breaks = []
upperBound = []
lowerBound = []
oosdRMSE = []
oosRSq = []

j = 0

### removing nan from matrix
variable = 3  #choose the variable that you want to examine
oneVar = pd.concat([df.iloc[30:noofObs,1],df.iloc[30:noofObs,variable]],axis = 1)
oneVar  = oneVar.dropna(how = 'any')
oneVar = oneVar.reset_index(drop=True)
X = oneVar.iloc[:,1]
Y = oneVar.iloc[:,0]
noOfObs = len(oneVar)

# Out of Sample
j = 0
while j < 34:

   XIS = X[j:60+j]
   YIS = Y[j:60+j]


   HM = np.mean(YIS)
   coef = coefficients(XIS,YIS)

   errorOLSSq.append(np.square(Y[60+j] - coef[0] - coef[1] * XIS.iloc[-1]))

   errorHMSq.append(np.square(Y[60+j] - HM))

   dESS.append((errorHMSq[j]) - (errorOLSSq[j]))

   breaks.append(coef[1])
   upperBound.append(coef[1] + coef[3]*2)
   lowerBound.append(coef[1] - coef[3]*2)
   
   j += 1

tempMSEOLS = np.mean(errorOLSSq)
tempMSEHM = np.mean(errorHMSq)

oosdRMSE.append(np.sqrt(tempMSEHM) - np.sqrt(tempMSEOLS))

oosRSq.append(1 - (np.sum(errorOLSSq)/np.sum(errorHMSq)))

###### In Sample

# Historical Mean
HMaverage = np.mean(Y)
InSampleErrorHM = Y-HMaverage

# OLS Model
olsCoeff = coefficients(X,Y)
residuals = Y - olsCoeff[0] - olsCoeff[1]*X
residuals = np.cumsum((residuals * residuals))
isRMSE = np.sqrt(np.mean(InSampleErrorHM)) - np.sqrt(np.mean(residuals))
InSampleErrorHM = np.cumsum((InSampleErrorHM * InSampleErrorHM))

cummuESSIS = InSampleErrorHM-residuals
cummuESSIS = [0] + list(cummuESSIS)
cummuESSoos = [0]*62
cummulative = np.cumsum(dESS)

i = 0

while i < len(dESS):
    
    cummuESSoos.append(cummulative[i])
    
    i+=1


DM = np.mean(dESS)/np.sqrt((1/len(dESS)*np.var(dESS)))

#plot graphs
#xAxis = np.linspace(1993,2017,len(cummuESSoos)-1)
#plt.plot(xAxis,X, linewidth=0.7, color = 'r')
#plt.ylabel('%s ' %(columnNames[variable]))
#plt.xlabel('Time')
#plt.title('Plot of %s' %(columnNames[variable]))
#plt.show()

xAxis = np.linspace(1993,2017,len(cummuESSoos))
fig = plt.figure()
bx = fig.add_subplot(111)
bx.axhline(y=0, color = 'r', linestyle='dashed', linewidth=0.5)
bx.plot(xAxis,cummuESSoos,linewidth=0.75, label='Out of Sample')
bx.plot(xAxis,cummuESSIS,linewidth=0.75, label='In Sample')
legend = bx.legend(loc='upper right', shadow=False, fontsize='small')
bx.set_xlabel('Time')
bx.set_ylabel('Cummulative ESS Difference')
bx.set_title('Cummulative delta ESS of %s' %(columnNames[variable]))
plt.show()
#
xCoeffAxis = np.linspace(2011.5,2017,len(breaks))
fig = plt.figure()
ax = fig.add_subplot(111)
coefficientLine = ax.plot(xCoeffAxis,breaks,linewidth=0.75, label='Coefficient')
upperBound = ax.plot(xCoeffAxis,upperBound,linewidth=0.75, label='Upper Bound',color = 'r')
ax.plot(xCoeffAxis,lowerBound,linewidth=0.75,label='Lower Bound', color = 'r')
lengend = ax.legend(loc='upper right', shadow=False, fontsize='small')
ax.set_xlabel('Time')
ax.set_ylabel('Coefficient')
ax.set_title('Beta Coefficient of %s Overtime' %(columnNames[variable]))
plt.show()
#
#regHAC = smf.ols('oneVar.iloc[:,0]~oneVar.iloc[:,1]',data=oneVar).fit(cov_type='HAC',cov_kwds={'maxlags':1})
#reg = smf.ols('oneVar.iloc[:,0]~oneVar.iloc[:,1]',data=oneVar).fit()
#print (regHAC.summary())
#print (reg.summary())
