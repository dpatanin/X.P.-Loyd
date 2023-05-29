from http.server import BaseHTTPRequestHandler, HTTPServer
import tensorflow as tf
import json

PORT = 8080

print("Loading AI...")
model: tf.keras.Sequential = tf.keras.models.load_model("ai-training\models\prototype-V1_terminal_19_05_2023 11_23_45.h5")
print("AI loaded")

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        print("received get request from ", self.client_address[0])
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        self.wfile.write(json.dumps({ "config":model.get_config()}).encode('utf-8'))

with HTTPServer(("", PORT), handler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()