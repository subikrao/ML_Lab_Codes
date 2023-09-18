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

def remove(s):
    if isinstance(s, str):
        return ' '.join(s.splitlines())
    else:
        return s
    
dtrain['newcol'] = dtrain['input'].apply(remove)
dtrain['EmbeddingsLM']=dtrain['newcol'].apply(lambda x:model.encode(str(x)))
print (dtrain['EmbeddingsLM'])
dtest['new'] = dtest['Equation'].apply(remove)
dtest['EmbeddingsLM']=dtest['new'].apply(lambda x:model.encode(x))
dtrain['classification'] = dtrain['output'].apply(lambda x: 'Correct' if x == 5 else ('Partially correct' if x in [2, 3, 4] else 'Incorrect'))

correct = dtrain[dtrain['classification'] == 'Correct']
partial = dtrain[dtrain['classification'] == 'Partially correct']
incorrect = dtrain[dtrain['classification'] == 'Incorrect']
correct_centroid = st.mean(correct['EmbeddingsLM'])
partial_centroid = st.mean(partial['EmbeddingsLM'])
incorrect_centroid = st.mean(incorrect['EmbeddingsLM'])

class_centroids = dtrain.groupby('classification')['input'].apply(lambda x: np.mean(x, axis=0))
class_spreads = dtrain.groupby('classification')['input'].apply(lambda x: np.std(x, axis=0))
