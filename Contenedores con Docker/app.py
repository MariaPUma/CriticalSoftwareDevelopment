from datetime import datetime
from flask import Flask, jsonify
from flask import request
from redis import Redis, RedisError
import os
import socket
import time
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model


# Connect to Redis
REDIS_HOST = os.getenv('REDIS_HOST', "localhost")
print("REDIS_HOST: "+REDIS_HOST)
redis = Redis(host=REDIS_HOST, db=0, socket_connect_timeout=2, socket_timeout=2)

app= Flask(__name__)

WindowSize = 3

# Leer el archivo
with open('UmbralDeAnomalias.txt', 'r') as file:
    umbral = float(file.readline().strip())  # Leer la primera línea

modelo = load_model("modelo.keras")

@app.route("/")
def hello():
    try:
        visits = redis.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"

    html ="""
        <h1>Práctica 2 - Desarrollo de Software Crítico</h1>
        <p><b>Hostname:</b> {hostname}</p>
        <p><b>Visits:</b> {visits}</p>
      

        <h2>3º de Ingeniería de Software</h2>
        
        <h2>Explicación de la Práctica</h2>
        <div class="explanation">
            En esta práctica de Desarrollo de Software Crítico, el objetivo es diseñar un sistema de monitorización de mediciones basado en contenedores utilizando Docker Swarm. El sistema incluye una API REST desarrollada con Flask para gestionar las mediciones y RedisTimeSeries como base de datos para el almacenamiento de datos temporales. Adicionalmente, se integrará Grafana para la visualización de las mediciones.
        </div>
        <b>
            Comandos : <br>
        </b>
        <div>
            <b>'/nuevo':</b> añadir nueva medición <br>
            <b>'/listar':</b> mostrar lista de mediciones<br>
            <b>'/eliminarLista':</b> elimina todas las mediciones<br>
        </div>
        
    """
    
    return html.format(hostname=socket.gethostname(), visits=visits)


@app.route("/nuevo")
def agregarNuevaMedicon():
    
    #Tomamos el datos de la ruta
    dato = request.args.get("dato")

    #Si existe un dato
    if  dato:
        try:
            #Transformamos el dato en un valor float
            float(dato)

            #Tomamos un timestamp actual y lo transformamos a milisegundos (formato adecuado)
            timestamp = int(time.time() * 1000)

            #Añadimos a la serie temporal
            redis.execute_command('TS.ADD','mediciones_lista',timestamp,dato)
            #redis.rpush("mediciones_lista",dato)

            #Imprimimos el dato insertado por pantalla
            html= "<h1>Práctica 2 - Desarrollo de Software Crítico</h1>" \
                            "<h2>Nuevo dato</h2>" \
                            "<b> Value:</b> {num}"
            return html.format(num=dato)

        except ValueError:
            return "<i>El valor introducido debe de ser un número</i>"
        except RedisError:
            return "<i>No se puede conectar a Redis</i>" 
    else:
        return "<b>El valor no es válido, por favor introduzca los datos de nuevo</b>"

    

    
@app.route("/listar")
def mostrarLista():

    try:
        #Mostramos el hostname
        hostname = socket.gethostname()
        fila = f"<p>Hostname: {hostname} </p>"

        #Si la serie temporal no existe mostramos un error
        if not redis.exists("mediciones_lista"):
            return "<h1>No hay datos en la serie</h1>"


        #Tomamos todas las medidas de la serie temporal
        mediciones = redis.execute_command('TS.RANGE','mediciones_lista', '-', '+')

        for medida in mediciones:
            #Transformamos el timestamp que está en milisegundos a un formato fecha con hora, minutos y segundos
            timestamp_legible = datetime.fromtimestamp(int(medida[0]) / 1000).strftime('%Y-%m-%d %H:%M:%S')
   
            #Comprobamos que el valor asociado al timestamp es un valor numérico
            valor = float(medida[1])
            
            #Añadimos a la lista la medición para mostrarla
            fila += f" <p>Medición, Timestamp: {timestamp_legible}, Valor: {valor}</p>"
            

        return fila
    except Exception as e:
        return e
    except RedisError:
        return "<i>No se puede conectar con Redis</i>"

    

@app.route("/eliminarLista")
def eliminarLista():
    try:
        #Si la serie temporal no existe mostramos un error
        if not redis.exists("mediciones_lista"):
            return "<h2>No hay datos en la serie</h2>"

        #Borramos la serie temporal
        redis.delete("mediciones_lista")
        return "<h2>La lista ha sido eliminada</h2>"
    except RedisError:
        return "<i>No se puede conectar con redis</i>"

    
@app.route("/detectar")
def deteccionAnomalias():

    #Tomamos el datos de la ruta
    dato = request.args.get("dato")

    #Si existe un dato
    if  dato:
        try:
            #Transformamos el dato en un valor float
            dato = float(dato)

            #Tomamos un timestamp actual y lo transformamos a milisegundos (formato adecuado)
            timestamp = int(time.time() * 1000)
            timestamp_legible = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')

            indice_actual=0
            
            if redis.exists("mediciones_lista"):
                #Para realizar la predicción consultamos los datos anteriores introducidos
                mediciones_lista = redis.execute_command('TS.RANGE','mediciones_lista', '-', '+')
                # Convertir las mediciones a un arreglo de NumPy
                mediciones_np = np.array(mediciones_lista)
                # Calcular el índice actual basado en la longitud de las mediciones
                indice_actual = len(mediciones_np)




            #Añadimos a la serie temporal
            redis.execute_command('TS.ADD','mediciones_lista',timestamp,dato)
            
                        
            anomalia = "no"
            prediccion = -1
            error = 0

            

            if indice_actual>=3:
                mediciones_np = mediciones_np[-WindowSize:]
                ventana_valores = [float(m[1]) for m in mediciones_np]
                ventana_valores = np.array(ventana_valores).reshape((1,WindowSize,1))
 
                prediccion = modelo.predict(ventana_valores)[0][0] 
                prediccion = prediccion[0]
                # Calcular la diferencia y verificar anomalía 
                error = abs(prediccion - dato) 
                error = error
                anomalia = "si" if error > umbral else "no"

                mediciones = [
                {
                    "time": datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S'), # Convertir a ISO 8601
                    "valor": valor
                }
                for ts, valor in mediciones_lista[-WindowSize:]
                ]

                

            else :
                mediciones={
                "medicion": timestamp_legible,
                "valor_real":dato
                }

            respuesta = {
            "mediciones": mediciones,
            "anomalia": anomalia,
            "prediccion":prediccion,
            "error":error
            }
            
            
            with open('Resultados.txt', 'a') as file:
                file.write(f"{respuesta}\n")


            #Imprimimos el dato nuevo por pantalla
            html= """
            <h1>Práctica 2 - Desarrollo de Software Crítico</h1>
            <h2>Nuevo dato</h2>
            <b> Value:</b> {num}<br>
            <b> Prediccion:</b> {pred}<br>
            <b> Error:</b> {err}<br>
            <b> Anomalia:</b> {anomal}<br>
            <b> Respuesta (almacenada en archivo .txt):</b> {res}<br>
            """
            return html.format(num=dato,pred= prediccion,err=error,anomal=anomalia,res= respuesta) 

        except ValueError:
            return "<i>El valor introducido debe de ser un número</i>"
        except RedisError:
            return "<i>No se puede conectar a Redis</i>" 
    else:
        return "<b>El valor no es válido, por favor introduzca los datos de nuevo</b>"




if __name__ == "__main__":
    PORT = os.getenv('PORT', 80)
    print("PORT: "+str(PORT))
    app.run(host='0.0.0.0', port=PORT)


