import pandas as pd
import numpy as np
from math import log2
import statistics as st
import matplotlib.pyplot as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
# import warnings
# warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

p=[]
entropy_cols=[]
IG=[]
for i in df.columns[0:-1]:
    print("Column:",i,":")
    count_i= list(df[i].value_counts())
    entropy_i=[]
    unique_elements=df[i].unique()
    entropy_j=[]
    for j in unique_elements:
        jyes = len(df_yes[df_yes[i]==j])
        jno = len(df_no[df_no[i]==j])
        # print("yes count for",j," = ",jyes)
        # print("no count for",j," = ",jno)
        jtotal = df[i].value_counts()[j]
        # print("total count for",j," = ",jtotal)
        p_yes= jyes / jtotal
        p_no= jno / jtotal
        if(p_no!=0 and p_yes!=0):
            if(p_no==1 and p_yes==1):
                ent_temp= 0
            else:
                ent_temp= -p_yes*np.log2(p_yes) - p_no*np.log2(p_no)
        elif(p_no!=0 and p_yes==0):
            if(p_no==1):
                ent_temp= 0
            else:
                ent_temp= -p_no*np.log2(p_no)
        elif(p_no==0 and p_yes!=0):
            if(p_yes==1):
                ent_temp= 0
            else:
                ent_temp= -p_yes*np.log2(p_yes)
        elif():
            ent_temp= 0
        # print("E(",j,") : ",ent_temp,sep='')
        entropy_j.append(ent_temp)
    for x in range(0,len(unique_elements)):
        el=unique_elements[x]
        valuee=(df[i].value_counts()[el]/length)*(entropy_j[x])
        entropy_i.append(valuee)
        
    print("\nentropy =",sum(entropy_i),sep='')
    entropy_cols.append(sum(entropy_i))
    IG_i = entropy_buys-sum(entropy_i)
    print("information gain = ",IG_i,"\n",sep='')
    IG.append(IG_i)

col_maxig = IG.index(max(IG))
colname=df.columns[col_maxig]