"""Static server over the repo root + POST /result collector (for browser auto-tests).

Results are appended to <out> as JSON lines.
Usage: python tools/collect_server.py [port] [out_path]
"""
import http.server
import json
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8788
OUT = sys.argv[2] if len(sys.argv) > 2 else r"E:\tmp\webnn_result.jsonl"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def do_POST(self):
        if self.path != "/result":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(body.decode("utf-8") + "\n")
        print("RESULT:", body.decode("utf-8"), flush=True)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


print(f"serving {ROOT} on 127.0.0.1:{PORT}, results -> {OUT}", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
