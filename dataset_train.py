import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os


file = open("/content/Autocomplete_Medical_Text_Project/data/surgery.txt", "r", encoding = "utf8")

# store data file in list
lines = []
for i in file:
    lines.append(i)

# Convert list to string
data = ""
for i in lines:
  data = ' '. join(lines) 

#replace unnecessary stuff with space
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')  #new line, carriage return, unicode character --> replace by space

#remove unnecessary spaces 
data = data.split()
data = ' '.join(data)
#data[:500]

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function
pickle.dump(tokenizer, open('/content/Autocomplete_Medical_Text_Project/token.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]
#sequence_data[:15]

vocab_size = len(tokenizer.word_index) + 1

sequences = []  

for i in range(3, len(sequence_data)):  #taking 4 words from which we will train... 
    words = sequence_data[i-3:i+1]      #... the prediction of 4th word for the first 3 words
    sequences.append(words)
    
#print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)


X = [] 
y = []

for i in sequences:
    X.append(i[0:3])  #first 3 words sequence is our X
    y.append(i[3])    #our resulting 4th word is our y which will be the output
    
X = np.array(X)
y = np.array(y)

y = to_categorical(y, num_classes=vocab_size)  #class vector to binary class matrix 
                                               #because later we use loss function as categorical crossentropy,
#y[:5]

