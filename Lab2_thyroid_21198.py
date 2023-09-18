import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as mp
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
import math
import seaborn as sns

excel_file = '19CSE305_LabData_Set3.1.xlsx'
df = pd.read_excel(excel_file, sheet_name='thyroid0387_UCI')

'''
A1: DATA EXPLORATION
->numeric values: age, TSH, T3, TT4, T4U, FTI, TBG
->nominal values that require one-hot encoding: 
sex on thyroxine, query on thyroxine, on antithyroid medication, sick, pregnant, thyroid surgery, I131 treatment, query hypothyroid, query hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH measured T3 measured, TT4 measured, T4U measured, FTI measured, TBG measured
->Outliers are present in TSH, T3, TT4, FTI, TBG
->Outliers are not there in T4U
->Calculating range of the numeric values:
'''

print("\nRange of the numeric values:\n")
for a in df.columns:
    if(a=='age' or a=='TSH' or a=='T3' or a=='TT4' or a=='T4U' or a=='FTI' or a=='TBG'):
        # aname=a+""
        removed_missing = df[df[a] != '?'] 
        max = removed_missing[a].max()
        min = removed_missing[a].min()
        print("Range of ",a," = (",round(min,5)," , ",round(max,5),")",sep="")

print("\nMean and variance of the numeric values:\n")
for b in df.columns:
    if(b=='age' or b=='TSH' or b=='T3' or b=='TT4' or b=='T4U' or b=='FTI' or b=='TBG'):
        bname=b+""
        removed = df[df[bname] != '?'] 
        print("->Mean of",b,"=",round(st.mean(removed[bname]),5))
        print("  Variance of",b,"=",round(st.variance(removed[bname]),5))

'''
A2: DATA IMPUTATION
-> Median may be employed for attributes which are numeric and contain outliers. TSH contains outliers.
-> Mean may be used when the attribute is numeric with no outliers
-> Mode may be employed for categorical attributes
#Ordinal values: referral source, Condition
'''
print("\nReplacing missing values:")
print("Median - Numeric attributes with outliers: TSH, T3, TT4, FTI, TBG")
print("Mean - Numeric attributes with no outliers: T4U")
print("Mode - Categorical attributes")
for col in df.columns:
    #outliers present in TSH, T3, TT4, FTI, TBG
    if(col=="TSH" or col=="T3" or col=="TT4" or col=="FTI" or col=="TBG"):
        removed = df[df[col] != '?'] 
        median = st.median(removed[col])
        df[col] = df[col].replace('?', median)
    elif(col=="T4U"):
        #no outliers found in T4U hence mean should be used
        T4Uremoved = df[df['T4U'] != '?'] 
        T4Umean = round((st.mean(T4Uremoved['T4U'])),2)
        df['T4U'] = df['T4U'].replace('?', T4Umean)
    else:
        #Handling missing nominal values using mode: 
        df[col] = df[col].replace('?', st.mode(df[col]))



# df.to_excel('updated_dataset.xlsx')

'''
A3: DATA NORMALIZATION / SCALING
->Min-max normalization is greatly affected by outliers
so it is preferred when there isn't any outlier in the data such as T4U
->Z-Score normalization is less affected by outliers
TSH, T3, TT4, FTI, TBG can be normalized using Z-Score
'''
# Min-max : (Aji - Aj_min)/(Aj_max - Aj_min)
df_norm = df.copy()
i='T4U'
df_norm[i] = (df_norm[i] - df_norm[i].min())/(df_norm[i].max() - df_norm[i].min())    
# Z-Score: Zij = (Aij - Aj mean)/Aj standard deviation
for j in df_norm.columns:
    if(j=="TSH" or j=="T3" or j=="TT4" or j=="FTI" or j=="TBG"):
        jcolumn=j+""
        df_norm[j]= (df_norm[j] - st.mean(df_norm[jcolumn])) / st.stdev(df_norm[jcolumn])
# print("\nNORMALIZED DATASET:\n",df_norm)
df_norm.to_excel('zscored_dataset.xlsx')

'''
A4: SIMILARITY MEASURE
Taking the first 2 observation vectors with only the binary attributes and replacing t=1 and f=0
Calculating the Jaccard Coefficient (JC)
JC = (f11) / (f01+ f10+ f11)
Calculating Simple Matching Coefficient (SMC) between the document vectors.
Use first vector for each document for this. Compare the values for JC and SMC and judge the
appropriateness of each of them.
SMC = (f11 + f00) / (f00 + f01 + f10 + f11)
'''
print("\nSimilary Measure:\n ")
binary_atts=['on thyroxine', 'query on thyroxine', 'on antithyroid medication' , 'sick' , 'pregnant' , 'thyroid surgery' , 'I131 treatment' , 'query hypothyroid' , 'query hyperthyroid' , 'lithium' , 'goitre' , 'tumor' , 'hypopituitary' , 'psych' , 'TSH measured' , 'T3 measured' , 'TT4 measured' , 'T4U measured' , 'FTI measured' , 'TBG measured']
binary_df=df.copy()
for d in binary_df.columns:
    if(d not in binary_atts):
        del binary_df[d]
    else:
        binary_df[d] = binary_df[d].replace('t',1)
        binary_df[d] = binary_df[d].replace('f',0)

# binary_df.to_excel('binary_attributes_only.xlsx')
row0 = binary_df.iloc[0]
row1 = binary_df.iloc[1]
f11=f01=f10=f00=0
for i in range (0, len(row0)):
    if(row0[i]==1 and row1[i]==1):
        f11+=1
    elif(row0[i]==0 and row1[i]==1):
        f01+=1
    elif(row0[i]==1 and row1[i]==0):
        f10+=1
    elif(row0[i]==0 and row1[i]==0):
        f00+=1

print(" f11 = ", f11)
print(" f10 = ", f10)
print(" f01 = ", f01)
print(" f00 = ", f00)
JC = f11/(f01+f10+f11)
print("\n Jaccard coefficient = ",JC)
SMC = (f11 + f00) / (f00 + f01 + f10 + f11)
print(" Simple matching coefficient = ",SMC)

#Ordinal values: referral source, Condition
#nominal value: sex one hot encoding
df_encoded=df.copy()
df_encoded['sex_M'] = np.where(df_encoded['sex'] == 'M', 1, 0)
df_encoded['sex_F'] = np.where(df_encoded['sex'] == 'F', 1, 0)
del df_encoded['sex']
del df_encoded['Record ID']
for q in df_encoded.columns:
    if(q in binary_atts):
        df_encoded[q] = df_encoded[q].replace('t',1)
        df_encoded[q] = df_encoded[q].replace('f',0)
# print(df_encoded)

sources = list(df_encoded['referral source'].unique())
#['other'=0, 'SVI'=1, 'SVHC'=2, 'STMW'=3, 'SVHD'=4, 'WEST'=5]
mapping = {'other': 0, 'SVI': 1, 'SVHC': 2, 'STMW': 3, 'SVHD': 4, 'WEST': 5}
df_encoded['referral source'] = df_encoded['referral source'].replace(mapping)

cond = list(df_encoded['Condition'].unique())
#['NO CONDITION', 'S', 'F', 'AK', 'R', 'I', 'M', 'N', 'G', 'K', 'A', 'KJ', 'L', 'MK', 'Q', 'J', 'C|I', 'O', 'LJ', 'H|K', 'D', 'GK', 'MI', 'P']
label_encoder = LabelEncoder()
df_encoded['Condition'] = label_encoder.fit_transform(df_encoded['Condition'])
# print(df_encoded)


'''
A5: COSINE SIMILARITY
'''
v1= np.array(df_encoded.iloc[0])
v2= np.array(df_encoded.iloc[1])
dprod = np.dot(v1, v2) 
sumv1=0
for i in v1:
    sumv1+=i*i
sumv2=0
for j in v2:
    sumv2+=j*j
v1mod = math.sqrt(sumv1)
v2mod = math.sqrt(sumv2)
cosine_sim = dprod / (v1mod * v2mod)
print(" Cosine similarity = ", cosine_sim)

'''
A6: HEATMAP
'''

data = df_encoded.iloc[0:20, :].values
datalength = len(data)

data_jc = binary_df.iloc[0:20, :].values
datajclen = len(data_jc)

jcmat = np.zeros((datajclen, datajclen))
smcmat = np.zeros((datalength, datalength))
cosmat = np.zeros((datalength, datalength))

for i in range(datajclen):
    for j in range(datajclen):
        if i != j:
            jcmat[i][j] = jaccard_score(data_jc[i], data_jc[j], average='binary')
            smcmat[i][j] = sum(np.logical_and(data_jc[i], data_jc[j])) / sum(np.logical_or(data_jc[i], data_jc[j]))

# sns.heatmap(jcmat)
# mp.tight_layout()
# mp.show()