from sklearn import preprocessing
from scipy.spatial import distance
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as mp
from sentence_transformers import SentenceTransformer
model=SentenceTransformer('sentence-transformers/sentence-t5-base')
a = 'training.xlsx'
dtrain = pd.read_excel(a)
y = 'testing.xlsx'
dtest = pd.read_excel(y)

    #class labelling
dtrain['class'] = dtrain['output'].apply(lambda x: 'Correct' if x == 5 else ('Partially correct' if x>=2 and x<5 else 'Incorrect'))
correct = dtrain[dtrain['class'] == 'Correct']
partial = dtrain[dtrain['class'] == 'Partially correct']
incorrect = dtrain[dtrain['class'] == 'Incorrect']

dtrain.to_excel('training_labelled.xlsx', index=False)

    #removing line breaks and converting to str
def remove(s):
    if isinstance(s, str):
        return ' '.join(s.splitlines())
    else:
        return s
    
dtrain['string_converted_input'] = dtrain['input'].apply(remove)
dtrain['EmbeddingsLM']=dtrain['string_converted_input'].apply(lambda x:model.encode(str(x)))

dtest['string_converted_equation'] = dtest['Equation'].apply(remove)
dtest['EmbeddingsLM']=dtest['string_converted_equation'].apply(lambda x:model.encode(str(x)))

    #stripping spaces in embeddings
dtrain['Embeddings_converted'] = dtrain['EmbeddingsLM'].apply(lambda x: x.tolist())
num_columns = len(dtrain['Embeddings_converted'][0]) 

for i in range(num_columns):
    col_name = "Embedding_"+str(i)
    dtrain[col_name] = dtrain['Embeddings_converted'].apply(lambda x: x[i])

dtrain.to_excel('training_data_updated.xlsx', index=False)

#taking out numeric values to calculate mean etc
numeric_dtrain=dtrain.copy()
numeric_dtrain.drop(columns=['input','string_converted_input'], inplace=True)

def labelenc(x):
    if(x=='Incorrect'):
        return 0
    elif(x=='Partially correct'):
        return 1
    if(x=='Correct'):
        return 2

numeric_dtrain['class_encoded'] = dtrain['class'].apply(labelenc)
numeric_dtrain.drop(columns=['class','EmbeddingsLM','Embeddings_converted'], inplace=True)

columns = numeric_dtrain.columns.tolist()
first_col = columns.pop(0)
columns.insert(-1, first_col)
numeric_dtrain = numeric_dtrain[columns]

numeric_dtrain.to_excel('numeric_data_updated.xlsx', index=False)

'''
A1. Evaluate the intra-class spread and interclass distances between the classes in your dataset. 
If your data deals with multiple classes, you can take any two classes. 

'''
class_0  = numeric_dtrain[numeric_dtrain['class_encoded'] == 0]
class_1  = numeric_dtrain[numeric_dtrain['class_encoded'] == 1]
class_2  = numeric_dtrain[numeric_dtrain['class_encoded'] == 2]
#selecting Correct class and Incorrect class
avg_vector_0 = class_0.mean().values
avg_vector_2 = class_2.mean().values
std_vector_0 = (class_0.std().values).tolist()
std_vector_2 = (class_2.std().values).tolist()

distance = np.linalg.norm(avg_vector_0 - avg_vector_2)
print(distance)

'''
A2. Take any feature from your dataset. 
Observe the density pattern for that feature by plotting the
histogram. Use buckets (data in ranges) for histogram generation and study. Calculate the mean and
variance from the available data.
(Suggestion: numpy.histogram()gives the histogram data. 
Plot of histogram may be achieved with matplotlib.pyplot.hist())
'''
feature = numeric_dtrain['output']
counts, bin_edges = np.histogram(feature, bins=10)
mp.hist(feature, bins=10, edgecolor='k')
mp.xlabel('Marks data buckets')
mp.ylabel('Frequency')
mp.title('marks feature histogram')
mp.show()
#Density pattern:
#it shows that most people scored between 2.5 to 3 and between 4.5 to 5
#there is narrow distribution between the buckets so the spread is less

'''
A3. Take any two feature vectors from your dataset. 
Calculate the Minkowski distance with r from 1 to 10. 
Make a plot of the distance and observe the nature of this graph.

'''
#choosing 360th and 1120th vectors.
feature_vector_1 = numeric_dtrain.iloc[359].values
feature_vector_2 = numeric_dtrain.iloc[1119].values

r = [1,2,3,4,5,6,7,8,9,10]
mink_dists = []
for i in range(1,11):
    d = np.linalg.norm(feature_vector_1 - feature_vector_2, ord=i)
    mink_dists.append(d)

mp.plot(r, mink_dists,marker='o')
mp.xticks([0,1,2,3,4,5,6,7,8,9,10])
mp.yticks([0,1,2,3,4,5,6,7,8,9,10])
mp.xlabel('r values')
mp.ylabel('minkowski distance')
mp.title('Plot of minkowski distances with r from 1 to 10: ')
mp.grid(True)
mp.show()