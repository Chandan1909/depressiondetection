from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib as plt
import pandas as pd

df = pd.read_csv('alphabet.csv')
dataset = df.values

x = dataset[:,0:20]
y = dataset[:,20]
y = to_categorical(y)

min_max_scaler = MinMaxScaler()
x_scale=min_max_scaler.fit_transform(x)

x_train, x_val_and_test, y_train, y_val_and_test= train_test_split(x_scale,y,test_size=0.2)
x_val,x_test,y_val,y_test=train_test_split(x_val_and_test,y_val_and_test,test_size=0.1)

model = Sequential()
model.add(Dense(units=32,activation='relu',input_dim=20))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=4,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
hist=model.fit(x_train,y_train,batch_size=10,epochs=100,validation_data=(x_val,y_val))
a=model.evaluate(x_test,y_test)[1]
b=model.predict(x_test)

print(a)
print(b)