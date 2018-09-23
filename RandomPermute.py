import numpy as np
from pandas import read_csv

def random(a, b):
    return np.random.randint(a, b)

dataframe = read_csv('sp500.csv')
data = np.array(dataframe)[:30]

block = 5

for i in reversed(range(1, len(data)//block)):
    if i is 1:
        break
    # Generate a random number from 1 to i-1 and swap data[i]
    # with data[random_number], for i = n-1 to 1
    k = random(1, i)
    print(i," ", k)
    
    # Swapping the blocks of data
    # Swapping expression a, b = b, a doesn't work
    for j in range(block):
        indx_a = (k-1)*block + j
        indx_b = (i-1)*block + j

        data[indx_a] = data[indx_a] + data[indx_b]
        data[indx_b] = data[indx_a] - data[indx_b]
        data[indx_a] = data[indx_a] - data[indx_b]
    
for i in range(len(data)):
    print(data[i])
    if (i+1)%5 == 0:
        print()