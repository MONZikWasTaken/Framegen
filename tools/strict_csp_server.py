# Serves the smoke page with a strict page CSP (no 'unsafe-eval' / 'wasm-unsafe-eval'): does the extension's wasm
# runtime still compile inside the content script? (The page CSP must not govern the extension's isolated world.)
import http.server, sys
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8768
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "media-src 'self' http://127.0.0.1:8765; img-src 'self' data:; connect-src 'self'; object-src 'none'")
        super().end_headers()
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
