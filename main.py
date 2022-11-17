import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os

from tensorflow.keras.models import load_model
#from colorama import Fore, Back, Style
import numpy as np
import pickle

# Load the model and tokenizer
model = load_model('/content/Autocomplete_Medical_Text_Project/model/next_words.h5')
tokenizer = pickle.load(open('/content/Autocomplete_Medical_Text_Project/token.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):

  sequence = tokenizer.texts_to_sequences([text])
  sequence = np.array(sequence)
  preds = np.argmax(model.predict(sequence))      #the best prediction is taken into consideration
  predicted_word = ""
  
  for key, value in tokenizer.word_index.items():
      if value == preds:
          predicted_word = key
          break
  
  print(predicted_word)   #the predicted word is printed
  return predicted_word


  while(True):      #this is the driver code 
  text = input("Enter your line: ")
  
  if text == "0":         #enter 0 to terminate the loop otherwise enter a sentence for predicting the next word
      print("Execution completed.....")
      break
  
  else:
      try:
          text = text.split(" ")
          text = text[-3:]
          st =""
          for i in range(len(text)):
            st+=text[i]+" "

          print(st)
          Predict_Next_Words(model, tokenizer, text)
          
      except Exception as e:
        print("Error occurred: ",e)
        continue