from kazoo.client import KazooClient
from kazoo.recipe.election import Election
import threading
import time
import random
import os
import requests 
import numpy as np
import uuid

# Importar el módulo signal
import signal

# Definir una función que se ejecuta cuando se recibe la señal de interrupción
def interrupt_handler(signal, frame):
    print(f"La aplicación {id} ha muerto")
    exit(0)

# Registrar la función como el manejador de la señal de interrupción
signal.signal(signal.SIGINT, interrupt_handler)

# Crear un identificador para la aplicación
id = os.getenv('ID',uuid.uuid4()) #tomamos variable del entorno
id = random.randint(1,100)


# Crear un cliente kazoo y conectarlo con el servidor zookeeper
ZOOKEEPER_HOST = os.getenv('ZOOKEEPER_HOST', 'zookeeper:2181') # para el compose es zookeeper:2181 #tomamos variable del entorno
client = KazooClient(hosts=ZOOKEEPER_HOST)
client.start()

# Crear una elección entre las aplicaciones y elegir un líder
election = Election(client, "/election",id)

# Definir una función que se ejecuta cuando una aplicación es elegida líder
def leader_func():
    while True:
        print('soy lider')
        time.sleep(3)

        # Obtener los hijos de /mediciones
        children = client.get_children("/mediciones")
        

        while len(children) == 0:
            time.sleep(2)
            children = client.get_children("/mediciones")
        

        try:
            mediciones = []  # Lista para almacenar las mediciones
            for name in children:
                # Obtén los datos del cliente
                (data, _) = client.get(f"/mediciones/{name}")
                
                # Decodifica y convierte a entero
                medicion = int(data.decode("utf-8")) if isinstance(data, bytes) else int(data)
                mediciones.append(medicion)  # Agrega los datos a la lista
        except ValueError as e:
            print(f"Error al convertir los datos a enteros: {e}")
        except Exception as e:
            print(f"Error al procesar las mediciones: {e}")

        # Calcula la media de las mediciones
        if mediciones:
            print(f"Conjunto de mediciones: {mediciones}")

            # Calcular la media de los valores
            media = np.mean(mediciones) 

            # Mostrar la media por consola
            print(f'La media de los valores de los hijos: {media}') 
        else:
            print("No se pudieron calcular las mediciones.")
    
        # Enviar la media usando requests
        url = 'http://web:80/nuevo' # Definimos la URL a la que queremos hacer la petición
        params = {'dato': media} # Definimos el parámetro que queremos enviar
        response = requests.get(url, params=params) # Hacemos la petición GET con el parámetro y guardamos la respuesta en una variable
        if response.status_code == 200: # Comprobamos si la petición fue exitosa
            print("La petición a la APP fue exitosa") # Imprimimos los datos
        time.sleep(5)


# Definir una función que se encarga de lanzar la parte de la elección
def election_func():
    # Participar en la elección con el identificador de la aplicación
    election.run(leader_func)

# Crear un hilo para ejecutar la función election_func
election_thread = threading.Thread(target=election_func, daemon=True)
# Iniciar el hilo
election_thread.start()

# Enviar periódicamente un valor a una subruta de /mediciones con el identificador de la aplicación
while True:
    # Generar una nueva medición aleatoria
    value = random.randint(75, 85)
    value = str(value)
    print(f'Id:  {id}, Value: {value}')

    # Esperar 5 segundos
    time.sleep(5)

    # Actualizar el valor de /values asociado al nodo

    try:
        # Asegurar que existe la ruta de acceso y si no la crea
        client.ensure_path("/mediciones")
        if client.exists(f"/mediciones/value{id}"):
            # Modificamos el nodo
            client.set(f"/mediciones/value{id}", value.encode("utf-8"))
        else:
            # Creamos un nodo con datos
            client.create(f"/mediciones/value{id}", value.encode("utf-8"), ephemeral=True)
        
        time.sleep(5)

    except:
        print('node creation exception (maybe exists)')
 