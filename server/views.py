from django.http import HttpResponse, HttpRequest
import tensorflow as tf
import pandas as pd
import numpy as np
import json

model: tf.keras.Sequential = tf.keras.models.load_model(
    "prototype-V1_terminal_19_05_2023 11_23_45.h5"
)
columns = [
    "progress",
    "open",
    "high",
    "low",
    "close",
    "closeNorm",
    "volume",
    "contracts",
    "entryPrice",
    "balance",
]


# Create your views here.
def index(request: HttpRequest):
    if request.method != "GET":
        return HttpResponse("Fuck off.")

    try:
        data = [json.loads(request.GET.get(c, "")) for c in columns]
        df = pd.DataFrame([*zip(*data)], columns=columns)
        return HttpResponse(model.predict(np.array([df.to_numpy()])))
    except Exception as e:
        print(e)
        return HttpResponse("Something went wrong.")
