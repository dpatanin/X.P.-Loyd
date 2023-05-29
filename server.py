from http.server import BaseHTTPRequestHandler, HTTPServer
import json

PORT = 8080

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        print("received get request from ", self.client_address[0])
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        self.wfile.write(json.dumps({ "name":"John", "age":30, "city":"New York"}).encode('utf-8'))

with HTTPServer(("", PORT), handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()