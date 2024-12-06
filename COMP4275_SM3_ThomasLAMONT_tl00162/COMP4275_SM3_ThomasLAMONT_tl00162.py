import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

df = pd.read_csv('FinancialPhraseBank.csv')
df = df[['Sentence', 'Sentiment']]

tokenizer = Tokenizer(num_words=5000) 
tokenizer.fit_on_texts(df['Sentence'])
X = tokenizer.texts_to_sequences(df['Sentence'])
X = pad_sequences(X, maxlen=100)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

l_voc = 5000
em_sz = 128
pad_sz = 100
filters = 128
kernel_size = 5 
pool_size = 2
latent_sz = 64 

finance_model = Sequential([
    Embedding(input_dim=l_voc, output_dim=em_sz, input_length=pad_sz),
    Dropout(0.25),
    Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', activation='relu', strides=1),
    MaxPooling1D(pool_size=pool_size),
    LSTM(latent_sz),
    Flatten(),
    Dense(3, activation='softmax') 
])

finance_model.compile(
    loss='categorical_crossentropy',  
    optimizer='adam',
    metrics=['accuracy']
)

finance_model.build((None, pad_sz))
finance_model.summary()

history_finance = finance_model.fit(
    X_train, 
    pd.get_dummies(y_train), 
    epochs=10,           
    batch_size=32,          
    validation_split=0.2   
)

results_finance = finance_model.evaluate(X_test, pd.get_dummies(y_test))
print("Model 1 - Accuracy:", results_finance[1], "Loss:", results_finance[0])


from tensorflow.keras.layers import GlobalAveragePooling1D

model2 = Sequential([
    Embedding(input_dim=l_voc, output_dim=em_sz, input_length=pad_sz),
    Dropout(0.5), 
    Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), 
    GlobalAveragePooling1D(),
    Dense(3, activation='softmax')
])

model2.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)

history2 = model2.fit(
    X_train, 
    pd.get_dummies(y_train),
    epochs=10,
    batch_size=64, 
    validation_split=0.2
)

results2 = model2.evaluate(X_test, pd.get_dummies(y_test))
print("Model 2 - Accuracy:", results2[1], "Loss:", results2[0])


from tensorflow.keras.regularizers import l2

model3 = Sequential([
    Embedding(input_dim=l_voc, output_dim=em_sz, input_length=pad_sz),
    Dropout(0.3),
    Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', strides=1, kernel_regularizer=l2(0.01)),
    MaxPooling1D(pool_size=2),
    LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01)),
    LSTM(16, dropout=0.3, kernel_regularizer=l2(0.01)),
    Dense(3, activation='softmax')
])

model3.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)

history3 = model3.fit(
    X_train, 
    pd.get_dummies(y_train), 
    epochs=10,            
    batch_size=32,          
    validation_split=0.2
)

results3 = model3.evaluate(X_test, pd.get_dummies(y_test))
print("Model 3 - Accuracy:", results3[1], "Loss:", results3[0])


plt.figure(figsize=(12, 6))

plt.plot(history_finance.history['accuracy'], label='Model 1 Train Accuracy')
plt.plot(history_finance.history['val_accuracy'], label='Model 1 Validation Accuracy')
plt.plot(history2.history['accuracy'], label='Model 2 Train Accuracy')
plt.plot(history2.history['val_accuracy'], label='Model 2 Validation Accuracy')
plt.plot(history3.history['accuracy'], label='Model 3 Train Accuracy')
plt.plot(history3.history['val_accuracy'], label='Model 3 Validation Accuracy')

plt.title('Training and Validation Accuracy for Models 1, 2, and 3')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))

plt.plot(history_finance.history['loss'], label='Model 1 Train Loss')
plt.plot(history_finance.history['val_loss'], label='Model 1 Validation Loss')
plt.plot(history2.history['loss'], label='Model 2 Train Loss')
plt.plot(history2.history['val_loss'], label='Model 2 Validation Loss')
plt.plot(history3.history['loss'], label='Model 3 Train Loss')
plt.plot(history3.history['val_loss'], label='Model 3 Validation Loss')

plt.title('Training and Validation Loss for Models 1, 2, and 3')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
