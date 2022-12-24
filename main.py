from tensorflow.keras.models import Sequential # Библиотека НС прямого распространения
from tensorflow.keras.layers import Dense # Библиотека НС Полносвязные слои
from tensorflow .keras import utils #Библиотека для преобразований данных
from google.colab import files #Библиотека для загрузки файлов
import numpy as np #Библиотека для работы с массивами
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


files.upload()

# считывание данных из файла YearPredictionMSD2.csv
data = pd.read_csv('YearPredictionMSD2.csv')
X = data.iloc[:,2:]
Y = data.iloc[:,1]


# считывание данных из файла YearPredictionMSD1.csv
# data = pd.read_csv('YearPredictionMSD1.csv')
# X = data.iloc[:,1:]
# Y = data.iloc[:,0]

# нормализация данных
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X = pd.DataFrame(X)

X_train, X_test, y_train, y_test =  train_test_split(X,Y, test_size=0.1)
print('Обучающие данные')
pd.concat([X_train,y_train], axis=1)

print('Тестовые данные')  
print(pd.concat([X_test,y_test], axis=1))

def build_model(numberNeurons1,numberNeurons2, optimizer, activation,size):
  model = Sequential()
  # Добавим к модели полносвязный слой с узлами(нейронами):
  model.add(Dense(numberNeurons1,input_shape=(X_train.shape[1],), activation=activation))
  for i in range(size):
    model.add(Dense(numberNeurons2, activation=activation))
  #исп. линейную ф-ию активации для выходного слоя, которая позволяет получать спектр зн-ий, а не только бинарный ответ
  model.add(Dense(1, activation='linear'))
  #производим компиляцию нашей модели
  model.compile(optimizer=optimizer, loss='mse',  metrics=['mae']) 

  return model




def create_plot(history):
  plt.title('Loss / Mean Squared Error')
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.show()

model = build_model(1000,500,"adam",'relu',1)
print(model.summary()) #вывод структуры НС (кол-во слоев, сколько там нейронов, кол-во обучаемых параметров-весовые коеффициенты НС)

#обучение модели
history = model.fit(X_train, y_train, epochs=40, batch_size=200,   validation_split=0.1,  verbose=1) #визуализация хода обучения
print(create_plot(history))

# Оценка качества
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)
print('Train: ',  train_mse, 'Test: ',test_mse)

pred = model.predict(X_test).flatten() 
for i in range(500):
   print("вычисленное значение НС: ", round(pred[i],3), ", верный результат: ", 
         round(y_test.iloc[i],3), ", разница: ", round(y_test.iloc[i] - pred[i],3))

#сохраняем НС в файл
model.save("NS.h5")
files.download("NS.h5")

#ипользование уже обученной НС
from keras.models import load_model
files.upload()
model = load_model("NS.h5")

#Подбор гиперпараметров
NumbersNeurons1=[1500,800,100]
NumbersNeurons2=[1000,600,100]
MasOptimizer=["adam","rmsprop","rmsprop"]
MasActiv=["relu","sigmoid","sigmoid"]
MaskolSL=[3,1,5]

kolepochs=[40,60,25]
kolbatch_size=[300,400,200]

result = pd.DataFrame([], columns=["numberNeurons1","numberNeurons2 - внут. слои", "optimizer","activation","kolSLoev", "kolepochs","kolbatch_size",
                                   "Train loss && MAE", "Test loss && MAE"])
i=0
for i in range(3):
  model = build_model(NumbersNeurons1[i],NumbersNeurons2[i],MasOptimizer[i],MasActiv[i], MaskolSL[i])
  model.fit(X_train, y_train, epochs=kolepochs[i], batch_size=kolbatch_size[i], validation_split=0.1,  verbose=0)
  train_mse1 = model.evaluate(X_train, y_train, verbose=0)
  test_mse1 = model.evaluate(X_test, y_test, verbose=0)
  result.loc[i]=[NumbersNeurons1[i],NumbersNeurons2[i],MasOptimizer[i],MasActiv[i], MaskolSL[i],kolepochs[i],kolbatch_size[i], train_mse1,test_mse1]


print("\n Таблица подбора гиперпараметров \n")
print(result.to_markdown())