import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.regularizers import l2

from keras import models
from keras import layers

from keras.datasets import imdb #데이터셋, 수치화 ->vector, sigmoid

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)


#단어를 텍스트로 디코딩
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value,key) for (key, value) in word_index.items()]
)
decode_review = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]]
)

#데이터 vector 변환
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    print(results.shape)
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 0, 1, 스칼라 값으로 저장되어 있음
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = Sequential()
model.add(Dense(16,activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, kernel_regularizer=l2(0.01), activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data=(x_val, y_val))


 #그리프 도식화  -> 과적합 발생
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
