# %% [markdown]
# # Práctica 1- Desarrollo de Software Crítico
# ## Detección de Anomalías con Machine Learning
# ---

# %% [markdown]
# En el siguiente archivo se muestra la memoria de la práctica 1, donde utilizaremos técnicas de Machine Learning para identidicar anomalías. Está contiene aclaraciones y las pruebas del código de las distintas partes de la práctica. 

# %% [markdown]
# Inicialmente, importaremos todos los paquetes necesarios para el desarrollo de los programas.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib


# %% [markdown]
# Procedemos a cargar los datos del fichero "datos.csv", que contiene los datos de las temperaturas registradas en un dispositivo industrial.

# %%
df = pd.read_csv("datos.csv",index_col=0,parse_dates=True)

print(df)

# %% [markdown]
# Mostramos gráficamente los datos del archivo
# 

# %%
df.plot()
plt.show()

# %% [markdown]
# Una vez cargados los datos procedemos a construir el array que contiene las ventanas, además del array de predicciones (el siguiente valor después de cada ventana).
# 
# ### Opción 1: `sliding_window_view`

# %%
# Crear las ventanas temporales
# Elegimos 3 como tamaño de ventana
n_steps = 3

# Convertimos el dataframe a un array de Numpy antes de usar sliding_window_view
raw_values = df.values.flatten() 

print(raw_values)
# Crear ventanas deslizantes usando sliding_window_view
windows = np.lib.stride_tricks.sliding_window_view(raw_values, window_shape=n_steps)

# Extraer X (ventanas) e y (valores a predecir)
X = np.array(windows[:-1])  # Todas las ventanas menos la última
y = np.array(raw_values[n_steps:])  # Los valores que predices (el siguiente valor después de cada ventana)

print('Array de ventanas : \n')
print(X)
print('\nArray de predicciones : \n')
print(y)

# %% [markdown]
# ### Opción 2: `split_sequence`
# También podemos utilizar la finción auxiliar `split_sequence` para crear ambos arrays.

# %%
# Función para dividir la secuencia en ventanas de n_steps
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Determina el índice final de la ventana
        end_ix = i + n_steps
        # Si hemos sobrepasado la longitud de la secuencia, terminamos
        if end_ix > len(sequence) - 1:
            break
        # Partes de la ventana (X) y la salida (y)
        seq_x = sequence[i:end_ix]
        seq_y = sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)  # Usa np.array() en lugar de array()

# %% [markdown]
# Como podemos observar obtenemos los mismos resultados que utilizando `sliding_window_view`.

# %%
# Crear las ventanas temporales
# Elegimos 3 como tamaño de ventana
n_steps = 3

# Convertir el dataframe a un array de Numpy antes de usar split_sequence
raw_values = df.values.flatten()  # Convertir el DataFrame a un array Numpy

print(raw_values)
# Crear ventanas deslizantes usando split_sequence
# Extraer X (ventanas) e y (valores a predecir)
X,y= split_sequence(raw_values,n_steps=n_steps)

print('Array de ventanas : \n',X)
print('\nArray de predicciones : \n',y)

# %% [markdown]
# ### Opcional : Dividir el conjunto de datos

# %% [markdown]
# Podemos dividir el conjunto de datos en datos de aprendizaje y de validación, 70% y 30% respectivamente. Esto con el objetivo de  garantizar que el modelo no solo funcione bien con los datos con los que se entrena, sino que también sea capaz de generalizar bien a datos nuevos y no vistos.
# 
# Sin embargo en esta práctica no lo implementaremos .

# %%
# Procedemos a dividir los datos en 70% de datos de entrenamiento y 30% de datos de validación
# Obtenemos el número total de datos
n_samples = X.shape[0]
percent_training = 0.7

# Calculamos el total de elementos que habrá en el conjunto de training
total_training = int(n_samples * percent_training)

# Permutamos aleatoriamente los índices
indices = np.random.permutation(n_samples)

# Dividimos los datos en entrenamiento y validación usando los índices permutados
train_indices = indices[:total_training]
val_indices = indices[total_training:]

X_train, X_val = X[train_indices], X[val_indices]
y_train, y_val = y[train_indices], y[val_indices]

print("Datos de entrenamiento:", X_train.shape)
print("Datos de validación:", X_val.shape)

# %% [markdown]
# ## Código LSTM
# 
# En el siguiente apartado se mostrarán los pasos del modelo LSTM

# %% [markdown]
# Redimensionamos el conjunto de datos, ya que la red neuronal **LSTM** espera 3 dimensiones: número muestras, pasos temporales, número features.

# %%
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# %% [markdown]
# ### Creamos el modelo LSTM
# Procedemos a crear el modelo de la red neuronal además de entrenarla para el conjunto de datos.

# %%
# Creamos un modelo secuencial: secuencia de capas
model = Sequential()

# Dimensionando la capa de entrada de datos 
model.add(Input(shape=(n_steps, n_features)))

# Añadimos la capa LSTM con 50 neuronas y activación 'relu'
model.add(LSTM(50, activation='relu'))

# Añadimos una capa densa
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()

# %% [markdown]
# **CARACTERÍSTICAS RED NEURONAL:**
# 
# 
# **Capa Densa** : es un tipo de capa en una red neuronal donde cada neurona está conectada a todas las neuronas de la capa anterior. Es adecuado añadirla si queremos predecir solo un valor, como es este caso.
# 
# 
# 
# **Capa LSTM** :  
# - Número de neuronas 50 :tomamos el número de 50 neuronas para que la red tenga la capacidad de capturar patrones complejos en la secuencia. 
# - Función activación `relu`: Eleginos la función de activación `relu` para introducir la no linealidad en el modelo, reduciendo el problema del "desvanecimiento del gradiente" (actualización de los pesos muy lenta debido a la reducción del gradiente) y mejorar la eficiencia del entrenamiento.
# 
# 
# **Compilación del modelo**: 
# - Optimizador `adam`: ajusta los pesos
# - Función de perdida `mse` : se corresponde con el Error Cuadrático Medio
# 

# %% [markdown]
# ### Entrenamos el modelo 
# Para el entrenamiento del modelo procedemos a hacer inicialmente 200 iteraciones.
# 

# %%
model.fit(X, y, epochs=200)

# %% [markdown]
# ### Detección de Anomalías

# %% [markdown]
# En este apartado determinaremos el criterio para identificar anomalías y lo aplicaremos al fichero de datos para mostrarlo.

# %% [markdown]
# Tomaremos el criterio basado en la métrica del `Error Absoluto Medio`, qué funcionará como un umbral. Esto con el fin de diferenciar entre variaciones normales en los datos y comportamientos que realmente son inusuales o inesperados. El umbral actuará como una "línea de corte" y permitirá identificar puntos en los datos que se desvían significativamente de lo esperado.

# %% [markdown]
# En este caso, solo consideramos anomalías a los valores que difieren del valor predicho por más de tres veces el error absoluto medio. Para detectar nievles de desviación poco probables de ocurrir por casualidad.

# %% [markdown]
# Inicialmente predecimos los valores y los almacenamos en la variable "predicted_values". Además, calcularemos el MAE (Error Absoluto Medio).

# %%
windows_size=10

# Conjunto de predicciones
predicted_values = model.predict(X).flatten()

# Calculamos el Error Absoluto Medio
mae = np.mean(np.abs(y.flatten()-predicted_values))
print(f"Error Absoluto Medio: {mae}")
print("Array de predicciones y array de valores reales")
print(predicted_values, predicted_values.shape)
print(y,y.shape)


# %% [markdown]
# Creamos el array con todas las fechas para poder iterar entre las variables y generar la gráfica.

# %%
fechas_test=df[windows_size:].index.to_numpy()  # Array con fechas de los datos de test
print(fechas_test,fechas_test.shape)

# %% [markdown]
# Creamos el array con todo el conjunto de anomalías detectadas con el criterio anterior. Posteriormente, mostramos en un gráfico donde mostramos las anomalías con puntos rojos, junto al conjunto de datos y el conjunto de datos predichos en naranja.

# %%
# Construimos el array con el conjunto de anomalías 
anomalies = np.array([True if np.abs(y[i]-predicted_values[i])>mae*3 else False for i in range(len(fechas_test))], dtype=bool)
y_test = df["value"][windows_size:]

# Mostramos los valores anómalos
print("El número de anomalias es %d sobre %s" %(np.sum(anomalies),anomalies.shape))
print(y_test[anomalies])

# Construimos la gráfica
plt.plot(fechas_test,y_test,color='blue',label='y_test')

plt.scatter(x=fechas_test, y=y_test, c='red', alpha=anomalies.astype(int),s=50)

predicted_values_new = predicted_values[:len(fechas_test)]
plt.plot(fechas_test, predicted_values_new, linestyle='--', linewidth=0.5, color='orange', label= "y_pred")


plt.legend()
plt.show()

# %% [markdown]
# El objetivo de este partado es detectar las anomalías para entender las causas subyacentes de los errores, y por consecuente poder mejorar la función del modelo.

# %% [markdown]
# ### Mejora del modelo

# %% [markdown]
# Tras el entrenamiento del modelo, observamos que los valores de la función de pérdida tienden a acercarse, en su mayoría, al 0 . Esto nos lleva a concluir que las predicciones del modelo son bastante exactas, ya que se acercan considerablemente a los valores reales.

# %% [markdown]
# Sin embargo, probaremos algunas medidas para observar si el modelo puede desempeñar una mejor función.

# %% [markdown]
# Para poder mejorar el rendimiento del sistema probaremos las siguientes opciones:
# 1. Añadir una capa más al modelo LSTM.
# 2. Normalizar los datos con el `MinMaxScaler()` para llevar los valores a un rango específico.
# 3. Modificar el número de épocas.
# 4. Añadir tamaños de lote
# 5. Aumentar el tamaño de ventana 
# 6. Ajuste de hiperparámetros

# %% [markdown]
# #### 1. Nueva Capa de neuronas

# %% [markdown]
# En esta sección, mejoraremos el modelo agregando una nueva capa neuronal. Esto nos permitirá evaluar si la adición de una capa adicional contribuye a mejorar el rendimiento y precisión del modelo en el proceso de aprendizaje.
# 

# %%
# Creamos un modelo secuencial: secuencia de capas
modelLSTMplus = Sequential()

# Dimensionando la capa de entrada de datos 
modelLSTMplus.add(Input(shape=(n_steps, n_features)))

# Añadimos una capa LSTM con 50 neuronas y activación 'relu'
modelLSTMplus.add(LSTM(50, activation='relu', return_sequences = True))

# Añadimos la capa LSTM con 50 neuronas y activación 'relu'
modelLSTMplus.add(LSTM(50, activation='relu'))

# Añadimos una capa densa
modelLSTMplus.add(Dense(1))

modelLSTMplus.compile(optimizer='adam', loss='mse')

modelLSTMplus.summary()

modelLSTMplus.fit(X, y, epochs=200)


# %%
# Calculamos el Error Absoluto Medio
predicted_values_mejoraCapa= modelLSTMplus.predict(X).flatten()

mae = np.mean(np.abs(y-predicted_values_mejoraCapa))
print("Error Absoluto Medio: ",mae)


# %% [markdown]
# > Conclusión: La adición de una capa adicional al modelo LSTM muestra mejora en el rendimiento, obteniendo un Error Absoluto Medio (MAE) de aproximadamente 0.70, reduciendo ligeramente el MAE original (0.7093). Esto sugiere que aumentar la profundidad del modelo contribuye a reducir el error, aunque en menor medida.
# 

# %% [markdown]
# Añadimos la mejora al modelo original.

# %% [markdown]
# #### 2. Normalización de los datos

# %% [markdown]
# En esta sección, implementaremos la normalización de los datos de entrada. Esta mejora tiene como objetivo optimizar el rendimiento del modelo al facilitar su proceso de aprendizaje y permitir una convergencia más rápida. Esta normalización de los datos la realizaremos con la función `MinMaxScaler()` (transforma las características de un conjunto de datos de forma que todos los valores estén dentro de un rango específico, generalmente entre 0 y 1).

# %%
# Normalizar los datos
scaler = MinMaxScaler() 
scaler_array = np.array(raw_values).reshape(-1, 1)
raw_seq_scaled = scaler.fit_transform(scaler_array)

# Crear las ventanas temporales
# Elegimos 3 como tamaño de ventana
n_steps = 3

# Convertimos el dataframe a un array de Numpy antes de usar sliding_window_view
raw_values = df.values.flatten() 

print(raw_values)
# Crear ventanas deslizantes usando sliding_window_view
windows = np.lib.stride_tricks.sliding_window_view(raw_seq_scaled.reshape(-1), window_shape=(n_steps))

# Extraer X (ventanas) e y (valores a predecir)
X = np.array(windows[:-1])  # Todas las ventanas menos la última
y = np.array(raw_values[n_steps:])  # Los valores que predices (el siguiente valor después de cada ventana)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Creamos un modelo secuencial: secuencia de capas
model = Sequential()

# Dimensionando la capa de entrada de datos 
model.add(Input(shape=(n_steps, n_features)))

# Añadimos una capa LSTM con 50 neuronas y activación 'relu'
model.add(LSTM(50, activation='relu', return_sequences = True))

# Añadimos la capa LSTM con 50 neuronas y activación 'relu'
model.add(LSTM(50, activation='relu'))

# Añadimos una capa densa
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()

# Entrenamos sobre el modelo que tiene solo una capa
model.fit(X, y, epochs=200)

# %%
print(raw_seq_scaled)

# %%
# Calculamos el Error Absoluto Medio
predicted_values_mejoraNormalizacion= model.predict(X).flatten()

mae = np.mean(np.abs(y-predicted_values_mejoraNormalizacion))
print("Error Absoluto Medio:",mae)

# %% [markdown]
# > Conclusión: La normalización de los datos no muestra una mejora significativa en el rendimiento, manteniendo un Error Absoluto Medio (MAE) de aproximadamente 0.773, similar al modelo anterior (alrededor de 0.709). Esto sugiere que, en este caso, la normalización no ofrece un beneficio en la precisión de las predicciones, lo cual puede deberse a que el modelo LSTM ya se adaptaba adecuadamente a la escala original de los datos o que se requerirá de ajustes adicionales en los hiperparámetros.

# %% [markdown]
# #### 3. Número de épocas

# %% [markdown]
# En esta sección, ajustaremos el número de épocas empleadas en el entrenamiento del modelo. El objetivo es analizar si existen diferencias significativas en el desempeño del modelo al aumentar o reducir este parámetro. Inicialmente, comenzaremos con 50 épocas y luego con 100.

# %% [markdown]
# Un número elevado de epochs puede permitir que el modelo aprenda patrones más complejos, pero si se elige un número muy alto, hay riesgo de sobreajuste (overfitting).

# %%
model.fit(X, y, epochs=50)

# %%
# Calculamos el Error Absoluto Medio
predicted_values_mejoraEpocas= model.predict(X).flatten()

mae = np.mean(np.abs(y-predicted_values_mejoraEpocas))
print("Error Absoluto Medio: ",mae)

# %%
model.fit(X, y, epochs=100)

# %%
# Calculamos el Error Absoluto Medio
predicted_values_mejoraEpocas2= model.predict(X).flatten()

mae = np.mean(np.abs(y-predicted_values_mejoraEpocas2))
print("Error Absoluto Medio: ",mae)

# %% [markdown]
# > Conclusión: La reducción del número de épocas a 50, ha incrementado el MAE a 0.723 y en el caso de las 100 épocas ha disminuido a 0,7095. En el primer caso, sugiere que el modelo no ha tenido suficientes iteraciones para aprender de manera óptima los patrones en los datos. En el segundo caso, mejora la media pero no muy signficativamente. 
# 
# >
# >Con respecto al modelo original, ninguno de las dos modificaciones de la época a reducido el MAE. Lo que nos indica que el rendimiento del sistema no se ve afectado significativamente por dichas variaciones.

# %% [markdown]
# #### 4. Tamaño por lote 

# %% [markdown]
# En esta sección, variaremos el tamaño del lote en la fase de aprendizaje. Inicialmente añadiremos un tamaño de lote (`batch_size`) de 32 y luego de 64. 
# 
# Este hiperparámetro indica cuántas muestras procesa el modelo antes de actualizar sus pesos durante el entrenamiento. Utilizaremos valores de batch_size que son potencias de 2, ya que están optimizados para el hardware moderno (particularmente en GPUs).  

# %% [markdown]
# ##### 1. Batch_size = 32

# %%
model.fit(X, y, epochs=200, batch_size=32)

# %%
# Calculamos el Error Absoluto Medio
predicted_values_mejoraBatch32= model.predict(X).flatten()

mae = np.mean(np.abs(y-predicted_values_mejoraBatch32))
print("Error Absoluto Medio:",mae)

# %% [markdown]
# ##### 2. Batch_size= 64

# %%
model.fit(X, y, epochs=200, batch_size=64)

# %%
# Calculamos el Error Absoluto Medio
predicted_values_mejoraBatch64= model.predict(X).flatten()

mae = np.mean(np.abs(y-predicted_values_mejoraBatch64))
print("Error Absoluto Medio: ",mae)

# %% [markdown]
# > Conclusión: Al aumentar el batch size a 32 y 64, el modelo obtuvo MAE de 0.696 y 0.706, respectivamente, comparado con el MAE de 0.709 sin batch size .  Esto puede deberse a que batch sizes menores suavizan las actualizaciones de los pesos, lo cual puede estabilizar el entrenamiento.
# 
# > Aunque no supone una gran mejora implementaremos al modelo un batch_size de 32.

# %% [markdown]
# #### 5. Aumento del tamaño de la ventana

# %% [markdown]
# En esta sección modificaremos el tamaño de la venta. Con el objetivo de observar si se altera la capacidad del modelo, atendiendo a su rendimiento y su tasa de error. Además de comprobar si se desempeña correctamente prediciendo cada 24h (ya que las fechas de los datos se establecen cada hora).

# %%
# Crear las ventanas temporales
# Elegimos 24 como tamaño de ventana
n_steps = 24

# Convertimos el dataframe a un array de Numpy antes de usar sliding_window_view
raw_values = df.values.flatten() 

print(raw_values)
# Crear ventanas deslizantes usando sliding_window_view
windows = np.lib.stride_tricks.sliding_window_view(raw_values, window_shape=n_steps)

# Extraer X (ventanas) e y (valores a predecir)
X = np.array(windows[:-1])  # Todas las ventanas menos la última
y = np.array(raw_values[n_steps:])  # Los valores que predices (el siguiente valor después de cada ventana)

print('Array de ventanas : \n')
print(X)
print('\nArray de predicciones : \n')
print(y)

# %%
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Creamos un modelo secuencial: secuencia de capas
model = Sequential()

# Dimensionando la capa de entrada de datos 
model.add(Input(shape=(n_steps, n_features)))

# Añadimos la capa LSTM con 50 neuronas y activación 'relu'
model.add(LSTM(50, activation='relu', return_sequences = True))

# Añadimos la capa LSTM con 50 neuronas y activación 'relu'
model.add(LSTM(50, activation='relu'))

# Añadimos una capa densa
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()
model.fit(X, y, epochs=200, batch_size=32)

# %%
# Calculamos el Error Absoluto Medio
predicted_values_mejoraTAMVentana= model.predict(X).flatten()

mae = np.mean(np.abs(y-predicted_values_mejoraTAMVentana))
print("Error Absoluto Medio:",mae)

# %% [markdown]
# > Conclusión: Tras el aprendizaje del modelo con el nuevo valor de ventana ,podemos observar que el modelo presentó un aumento en el tiempo de aprendizaje. Además de que el modelo presentó un aumento en el MAE de 0.709 a 0.739. Este resultado indica que estos ajustes no mejoraron la métrica y resultaron en un desempeño inferior al modelo original.

# %% [markdown]
# #### 6. Ajuste de hiperparámetros

# %% [markdown]
# En esta sección, optimizaremos el modelo modificando diversos hiperparámetros.
# 
# Modificaremos el número de neuronas en la capa LSTM y añadiremos una capa de Dropout del 0.2, para reducir el sobreajuste y mejorar la capacidad de generalización del modelo. 

# %%
# Creamos un modelo secuencial: secuencia de capas
model = Sequential()

# Dimensionando la capa de entrada de datos 
model.add(Input(shape=(n_steps, n_features)))

# Añadimos la capa LSTM con 50 neuronas y activación 'tanh'
model.add(LSTM(100, activation='relu', return_sequences = True))

# Añadimos la capa LSTM con 50 neuronas y activación 'tanh'
model.add(LSTM(100, activation='tanh'))

# Añadimos la capa
model.add(Dropout(0.2))

# Añadimos una capa densa
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(X, y, epochs=200,batch_size=32)

# %%
# Calculamos el Error Absoluto Medio
predicted_values_mejoraHiper= model.predict(X).flatten()

mae = np.mean(np.abs(y-predicted_values_mejoraHiper))
print(mae)

# %% [markdown]
# >Conclusión: Tras el aprendizaje del modelo actualizado,podemos observar que el modelo presentó un aumento en el MAE de 0.7017 a 3.48, junto al aumento del tiempo de aprendizaje. Este resultado indica que, aunque el modelo fue modificado para captar patrones más complejos con un mayor número de neuronas y una función de activación diferente, estos ajustes no mejoraron la métrica y resultaron en un desempeño inferior al modelo original.

# %% [markdown]
# #### Conclusión Mejora

# %% [markdown]
# En vista de los resultados, podemos concluir que el modelo resultante de las mejoras es el siguiente:

# %%
model = Sequential()

# Dimensionando la capa de entrada de datos 
model.add(Input(shape=(n_steps, n_features)))

# Añadimos la capa LSTM con 50 neuronas y activación 'tanh'
model.add(LSTM(50, activation='relu', return_sequences = True))

# Añadimos la capa LSTM con 50 neuronas y activación 'tanh'
model.add(LSTM(50, activation='relu'))

# Añadimos una capa densa
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(X, y, epochs=200,batch_size=32)

# %% [markdown]
# Este modelo tiene como Error Absoluto Medio: 0.696
# 
# Esta red neuronal LSTM es la que mejor se ajusta a los datos reales. Sus dos capas LSTM de 50 neuronas capturan patrones complejos, y la capa densa final permite predicciones precisas. Con los hiperparámetros, el modelo logra una convergencia rápida y minimiza errores. Y por último, las 200 épocas junto al tamaño del lote de 32 , aseguran una buena generalización sin sobreajuste. 
# 

# %% [markdown]
# ## Código Autoencoder:
# En el siguiente apartado se mostrarán los pasos del modelo Autoencoder

# %% [markdown]
# El modelo Autoencoder, es un tipo de red neuronal. Este modelo está compuesto por dos partes:
# - **Codificador** : Esta parte de la red transforma el dato de entrada a una representación de menor dimensión, conocida como el "espacio latente" o "código". El codificador actúa comprimiendo la información.
# 
# - **Decodificador** : Luego, el decodificador toma esa representación comprimida y trata de reconstruir el dato original. El decodificador trata de descomprimir la información para que se asemeje lo más posible a la entrada original.
# 
# 
# El propósito del modelo es minimizar la diferencia entre la entrada original y la salida reconstruida.

# %% [markdown]
# ### Creamos y entrenamos el modelo

# %% [markdown]
# Creamos el modelo y lo entrenamos con el conjunto de datos.

# %%
# Este paso podríamos evitarlo ya que está hecho en el código de la red neuronal LSTM
n_features =1
n_in = X.shape[1]
print(X)
sequence = X.reshape((X.shape[0], n_in, n_features))

modelAE = Sequential()
modelAE.add(Input(shape=(n_in, 1)))
modelAE.add(LSTM(100, activation='relu'))
modelAE.add(RepeatVector(n_in))
modelAE.add(LSTM(100, activation='relu', return_sequences=True))
modelAE.add(TimeDistributed(Dense(1)))
modelAE.compile(optimizer='adam', loss='mse')
modelAE.summary()
# fit model
modelAE.fit(sequence, sequence, epochs=100)
modelAE.save('modelo.keras')


# %% [markdown]
# **Capas LSTM** :  
# - La primera capa LSTM tiene 100 unidades esta es el codificador que procesa la secuencia y extrae la representación latente.
# - La segunda capa LSTM también tiene 100 unidades, pero en este caso devuelve secuencias completas, ya que estamos decodificando la información latente.
# 
# 
# 
# **RepeatVector()** :  
# - Este repite el estado comprimido (el vector latente) tantas veces como el tamaño original de la secuencia, para poder reconstruir la secuencia completa en la siguiente capa.
# 
# 
# 
# **TimeDistributed(Dense(1))** :  
# - Aplica una capa densa (neurona con salida lineal) sobre cada paso de la secuencia. La salida tiene la misma forma que la secuencia de entrada.
# 
# 
# 
# **Compilación del modelo**: 
# - Optimizador `adam`: ajusta los pesos
# - Función de perdida `mse` : se corresponde con el Error Cuadrático Medio
# 

# %% [markdown]
# Ya creado y entrenado el modelo procedemos a generar la predicción, es decir, tratar de recrear la secuencia de entrada a partir de lo que ha aprendido.

# %%
yhat = modelAE.predict(sequence, verbose=0)
print(yhat)
print(yhat.shape)
print(sequence.shape)



# %% [markdown]
# Obtenemos los últimos pasos de la secuencia

# %%
print(yhat[:,-1,0])

# %% [markdown]
# ### Detección de anomalías

# %% [markdown]
# Para la detección de anomalías utilizaremos tres veces el valor del Error Absoluto Medio como umbral.

# %%
# Calcular MAE
mae = np.mean(np.abs(raw_values[n_steps:] - yhat[:, -1, 0]))
print("Error Absoluto Medio: ",mae)

# Definir umbral de anomalía
umbral = mae * 3
print("Umbral: ",umbral)

# Identificar anomalías
residuals = np.abs(raw_values[n_steps:] - yhat[:, -1, 0])
anomalies = residuals > umbral

# Obtener índices de anomalías
anomaly_indices = np.where(anomalies)[0]

print(f"Número de anomalías : {anomaly_indices.size}")


# %%
with open('UmbralDeAnomalias.txt', 'w') as file:
    file.write(f"{umbral}\n")

# %% [markdown]
# Por último, construimos la gráfica indicando con puntos rojos las anomalías.

# %%
plt.figure(figsize=(20, 10))

# Graficar los datos
plt.plot(df.index[n_steps:], raw_values[n_steps:], label='Valores Reales', color='blue')

# Marcar anomalías
plt.scatter(df.index[anomaly_indices + n_steps], raw_values[anomaly_indices + n_steps], 
            color='red', marker='o', label='Anomalías', s=100)

plt.legend()
plt.title('Predicciones de autoencoder con ventanas de datos')
plt.xlabel('Fecha')
plt.ylabel('Valores')
plt.grid()
plt.show()

# %% [markdown]
# Como podemos observar el Error Cuadrático Medio tiene valor del 0.75 aproximadamente. Esto nos indica que el autoencoder ha aprendido a replicar la secuencia original de manera razonablemente precisa.

# %% [markdown]
# ## Código Isolation forest:
# En el siguiente apartado se mostrarán los pasos del modelo Isolation forest.

# %% [markdown]
# Creamos el modelo de Isolation forest según los datos anteriores. Este es un árbol de decisión especializado para aislar observaciones individuales, para la detección de anomalías.

# %%
X= df['value'].values.reshape(-1,1)
print(X)

# %% [markdown]
# Creamos una instancia del modelo Isolation Forest con un parámetro `contamination` de 0.01, lo que indica que se espera que aproximadamente el 1% de los datos que sean anomalías. Asimismo, se añade un `random_state` de 42, este hiperparámetro asegura que los resultados sean reproducibles. Finalmente entrenamos el modelo de datos.

# %%
# Entrenar el modelo de IsolationForest con los datos
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(X)

# Hacer predicciones con el modelo entrenado
y_pred = clf.predict(X)
print(y_pred)

# %% [markdown]
# Transformamos las anomalías al valor de 1 y las normales al 0, para facilitar el conteo de anomalías del conjunto de predicciones.

# %%
# Convertir la predicción a un formato que sea más fácil de manejar
# 1 indica anomalías, 0 indica normales
anomalies = np.where(y_pred == -1, 1, 0)
print(anomalies)
anomaly_indices = np.where(anomalies==1)[0]

print(f"El número de anomalías es de : {np.sum(anomalies)} / {anomalies.shape[0]}")




# %% [markdown]
# Por último, construimos la gráfica indicando con puntos rojos las anomalías.

# %%
# Graficar los datos
plt.figure(figsize=(20, 10))
plt.plot(df.index, raw_values, label='Valor', color='blue')
plt.scatter(df.index[anomaly_indices], raw_values[anomaly_indices], color='red', marker='o', label='Anomalías', s=100)

plt.legend()
plt.title('Predicciones de IsolationForest con ventanas de datos')
plt.xlabel('Fecha')
plt.ylabel('Valores')
plt.grid()
plt.show()

# %% [markdown]
# ## Conclusión de Modelos

# %% [markdown]
# En este apartado del documento, procederemos a observar las principales diferencias y puntos fuertes de los modelos realizados. 

# %% [markdown]
# ### Red neuronal LSTM
# 
# Este modelo presenta un **Error Absoluto Medio de 0.696, y sin mejoras 0.709**, el cual es un claro indicador de que el modelo es capaz de predecir los valores correctos con bastante precisión. Asimismo, es capaz de detectar **77 anomalías** sobre 7257. Las anomalías observadas en el gráfico, sugieren que el modelo es capaz de detectar los patrones anómalos de las predicciones además de los valores anómales puntuales.
# 
# Entre los puntos fuertes de dicho sistema, podemos destacar su capacidad de detectar patrones complejos en el conjunto de datos (como podemos observar con el conjunto de anomalías), y su propiedad de memoria a largo plazo, que le permite mantener la información relevante sobre el conjunto.
# 
# Los puntos débiles más destacables de este sistema, como hemos visto en la fase de mejora, son las dificultades que encontramos al entrenar los modelos, ya que requiere mucho tiempo de computo.
# 

# %% [markdown]
# ### Autoencoder
# 
# Este modelo presenta un **Error Absoluto Medio de 0.737**, aunque es un resultado mayor que el modelo de red neuronal LSTM, sugiere un buen funcionamiento en la detección de anomalías en el conjunto de datos. Asimismo, es capaz de detectar **72 anomalías** de 7257. Las anomalías observadas en el gráfico son similares al del modelo anterior, por tanto podemos concluir que tiene bastante precisión, sin embargo, es ligeramente inferior con respecto al anterior modelo.
# 
# Entre los puntos fuertes, destacamos que este modelo ha tenido un tiempo de entrenamiento destacablemente menor con respecto al anterior. Este modelo comparte los puntos fuertes del anterior modelo, debido a que presenta capas de red neuronal LSTM.
# 
# Con respecto a los puntos débiles, observamos que este modelo presenta una menor precisión, aunque está es mínima, con la diferencia de 77 y 72 anomalías detectadas.

# %% [markdown]
# ### Isolation Forest
# 
# Este modelo presenta **un número de anomalías igual a 73**, aunque este número se asemeje a los otros dos modelos, al contemplar la gráfica podemos observar como los valores anómalos son aquellos que son extremos, sugiriendo que el sistema es capaz de detectar las anomalías puntuales (aquellas que contienen valores anómalos) pero no las contextuales (patrones en el conjunto de datos inesperados). Además, la detección de anomalías depende estrechamente de la estimación que se establezca con respecto a las anomalías esperadas (parámetro: `contamination`).
# 
# De este sistema podemos destacar, su facilidad de computo, el tiempo requerido para el entrenamiento del modelo se reduce sustancialmente con respecto a los otros dos modelos.
# 
# Con respecto a los puntos débiles, además del factor de contaminación, se encuentra una mayor dificultad a la hora de interpretar los resultados de las anomalías detectadas.
# 

# %% [markdown]
# Tomando en cuenta todo lo anterior, podemos concluir que el modelo de red neuronal LSTM es el más eficiente en la detección de patrones y valores anómalos.


