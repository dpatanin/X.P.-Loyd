import tensorflow as tf
from tensorflow.python.client import device_lib

#* Use this file to run debug print scripts
spacer = "##################################################################"

print(spacer) 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(spacer)
print(spacer)
print(device_lib.list_local_devices())
