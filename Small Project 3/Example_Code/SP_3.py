# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# https://www.kaggle.com/code/mohameddanterr/sentiment-analysis-lstm-conv1d

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM,Dense,Bidirectional,Dropout,Conv1D,MaxPooling1D,Embedding,Flatten
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split





# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv('Mental_Health.csv')
print("df.head() ", df.head())
print("df.tail() ", df.tail())
print("df.info() ", df.info())
print("df.nunique()", df.nunique())
print("df.status.unique() ", df.status.unique())
print("df.status.value_counts() ", df.status.value_counts())

df = df[df.status != 'Bipolar']
df = df[df.status != 'Stress']
df = df[df.status != 'Personality disorder']

print("df.status.value_counts() ", df.status.value_counts())
sns.countplot(data = df , x = df.status)

print("df.isnull().sum() ", df.isnull().sum())

df.statement = df.statement.fillna(df.statement.mode()[0])
print("df.isnull().sum() ", df.isnull().sum())

print("df.columns ", df.columns)
df.drop('Unnamed: 0' , axis = 1,inplace = True)
print("df.columns ", df.columns)

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopwordss = stopwords.words('english')
print("stopwordss ", stopwordss)
df['status'].isnull().sum()

stopwordss = stopwords.words('english')
lem = WordNetLemmatizer()


def clean(line):
    line = line.lower()
    line = re.sub(r'\d+', '', line)
    line = re.sub(r'[^a-zA-Z0-9\s]', '', line)
    translator = str.maketrans('', '', string.punctuation)
    line = line.translate(translator)
    words = [word for word in line.split() if word not in stopwordss]
    #     words = [lem.lemmatize(word) for word in words]

    return ' '.join(words)

df['statement'] = df['statement'].apply(clean)
print("df.tail() ", df.tail())

for g in df['statement']:
    maxx = g.split()
    m = max([len(maxx)])

print("m ", m)

le = LabelEncoder()
df['status']=le.fit_transform(df['status'])
print(list(le.classes_))
print(le.transform(['Anxiety', 'Depression', 'Normal', 'Suicidal']))

# Model
x = df['statement']
y = df['status']

tokenizer = Tokenizer(oov_token='<unk>',num_words=2500)
tokenizer.fit_on_texts(x.values)
data_x = tokenizer.texts_to_sequences(x.values)

vocab = tokenizer.word_index
l_voc = len(vocab)
print(l_voc)

em_sz = 50
pad_sz = 42
latent_sz = 200
data_x = pad_sequences(data_x,maxlen=pad_sz,padding = 'post',truncating = 'post')

kernel_size = 5
filters = 64
pool_size = 4

model = Sequential()
model.add(Embedding(l_voc,em_sz,input_length = pad_sz))
model.add(Dropout(0.25))

model.add(Conv1D(filters,kernel_size,padding = 'valid',activation = 'relu',strides = 1))
# model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size = pool_size))
model.add(LSTM(latent_sz))
model.add(Flatten())
model.add(Dense(4,activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.build((None , pad_sz))
model.summary()

plot_model(model, show_shapes=True,show_layer_names=True,dpi = 90)

y= pd.get_dummies(y).values
x_train,x_test,y_train,y_test = train_test_split(data_x,y,random_state=0,shuffle = True,stratify=y , test_size = .2)
history = model.fit(x_train,y_train,batch_size=128, epochs=10,validation_data=(x_test,y_test))

score, acc = model.evaluate(x_test, y_test)
print(score)
print(acc)

p = model.predict(x_test)
print("p = ", p)

pred = np.argmax(model.predict(x_test[7:8]))
print("pred ", pred)
print("y_test[7:8] ", y_test[7:8])

prediction = le.inverse_transform(pred.reshape(1))[0]
print("prediction ", prediction)

history_dict = history.history
history_dict.keys()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training acc')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()

plt.show()















