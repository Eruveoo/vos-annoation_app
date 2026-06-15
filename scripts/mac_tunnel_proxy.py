#!/usr/bin/env python3
"""Forward http://127.0.0.1:PORT to a remote HTTPS API (e.g. localhost.run tunnel)."""
import ssl
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TARGET = sys.argv[1].rstrip("/")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 12212
SKIP_HEADERS = {"host", "connection", "transfer-encoding", "content-length"}
SSL_CTX = ssl.create_default_context()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        url = f"{TARGET}{self.path}"
        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in SKIP_HEADERS
        }
        req = urllib.request.Request(url, data=body, method=self.command, headers=headers)
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=3600) as resp:
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() not in SKIP_HEADERS:
                        self.send_header(key, val)
                self.end_headers()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except urllib.error.HTTPError as err:
            self.send_response(err.code)
            for key, val in err.headers.items():
                if key.lower() not in SKIP_HEADERS:
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(err.read())
        except Exception as exc:
            self.send_error(502, str(exc))

    def log_message(self, fmt, *args):
        print(f"[mac-tunnel-proxy] {args[0]}")

    do_GET = _forward
    do_POST = _forward
    do_PUT = _forward
    do_DELETE = _forward
    do_PATCH = _forward
    do_OPTIONS = _forward


def main():
    print(f"Forwarding http://127.0.0.1:{PORT} -> {TARGET}")
    print("Leave this running; frontend uses http://127.0.0.1:12212 (no .env changes).")
    ThreadingHTTPServer(("127.0.0.1", PORT), ProxyHandler).serve_forever()


if __name__ == "__main__":
    main()
