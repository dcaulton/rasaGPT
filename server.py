from http.server import HTTPServer, BaseHTTPRequestHandler


class CORSRequestHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        try:
            file_to_open = open(self.path[1:]).read()
            self.send_response(200)
        except FileNotFoundError:
            file_to_open = 'File not found'
            self.send_response(404)

        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(bytes(file_to_open, 'utf-8'))


def run(server_class=HTTPServer, handler_class=CORSRequestHandler, port=8000):
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


run()
