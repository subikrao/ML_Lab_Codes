import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
excel_file = 'Lab_Session1_Data.xlsx'
df = pd.read_excel(excel_file, sheet_name='Purchase data')
# print(df)
mat= df.iloc[0:11, 1:5]
# Ax=C
matrix_A = df.iloc[0:11, 1:4]
matrix_C = df.iloc[0:11, 4]
# matrix_A = np.array(matrix_A)

dimensionality_of_vector_space = matrix_A.ndim
print("dimensionality of the vector space = ",dimensionality_of_vector_space)
rank = np.linalg.matrix_rank(matrix_A)
print("rank of matrix A = ",rank)
pseudo_inverse = np.linalg.pinv(matrix_A)

# print(matrix_C)
print("Rank of the matrix = ", rank, "\n")
print ("Pseudo inverse matrix = ", pseudo_inverse, "\n")
matrix_X = np.dot(pseudo_inverse,matrix_C)
print("Matrix X\n", matrix_X, "\n")
df['Category'] = np.where(df['Payment (Rs)'] > 200, 'RICH', 'POOR')
print(df['Category'], "\n")
df.to_excel('Lab_Session1_Data_updated.xlsx', index=False)
df['Category_Encoded'] = df['Category'].apply(lambda x: 0 if x == 'POOR' else 1)
plt.xlabel('Payment (Rs)')
plt.ylabel('Category_Encoded')
plt.title('Sorted Vector V1')
plt.legend()
plt.grid()
plt.show()
#plot color is blue
# plt.plot(V1, color='red')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Sorted Vector V1')
# plt.show()