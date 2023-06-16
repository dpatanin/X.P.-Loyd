import tensorflow as tf
import keras as k

# on thread 1
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    k.backend.set_session(session)
    model = k.models.load_model("model/1/saved_model.pb")

# on thread 2
with session.graph.as_default():
    k.backend.set_session(session)
    model.predict(x, **kwargs)