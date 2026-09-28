#!/usr/bin/env python3
"""Static file server with the cross-origin isolation headers that
SharedArrayBuffer (and therefore the multithreaded femto wasm build)
requires. Usage:

    python3 serve.py [port] [--tls]     # default port 8000

then open http(s)://localhost:<port>/index.html. The single-threaded build
works with any static server; only the FEMTO_WASM_THREADS=ON build needs
these headers.

SharedArrayBuffer additionally requires a secure context, which plain http
only counts as on localhost: to open the page from another machine, pass
--tls to serve https with a self-signed certificate (generated once, into
~/.cache/femto-serve, so the key is never inside the served directory --
browsers show a warning to click through the first time)."""

import os
import ssl
import subprocess
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


class IsolatedHandler(SimpleHTTPRequestHandler):
    # keep-alive, so a browser reuses a handful of connections (one TLS
    # handshake each) instead of opening a new one for every worker script
    protocol_version = "HTTP/1.1"
    # an idle connection ties up its own thread for this long, and nothing else
    timeout = 30

    def handle(self):
        # The listening socket is wrapped with do_handshake_on_connect=False, so
        # the TLS handshake happens here, on this connection's thread. With the
        # default it runs inside accept(), on the single serving thread, where a
        # speculative connection that a browser opens and never speaks on stalls
        # every other client until it closes -- the threaded wasm module opens
        # dozens of workers at once, so pages took minutes to load over --tls.
        if isinstance(self.connection, ssl.SSLSocket):
            try:
                self.connection.do_handshake()
            except OSError:   # dropped preconnect, timeout, certificate probe
                return
        super().handle()

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def tls_context():
    certdir = os.path.expanduser("~/.cache/femto-serve")
    cert, key = os.path.join(certdir, "cert.pem"), os.path.join(certdir, "key.pem")
    if not (os.path.exists(cert) and os.path.exists(key)):
        os.makedirs(certdir, exist_ok=True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", key, "-out", cert, "-days", "365",
                        "-subj", "/CN=femto-serve"], check=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


if __name__ == "__main__":
    args = sys.argv[1:]
    tls = "--tls" in args
    if tls:
        args.remove("--tls")
    port = int(args[0]) if args else 8000
    server = ThreadingHTTPServer(("", port), IsolatedHandler)
    if tls:
        server.socket = tls_context().wrap_socket(server.socket, server_side=True,
                                                  do_handshake_on_connect=False)
    scheme = "https" if tls else "http"
    print(f"serving on {scheme}://localhost:{port} with COOP/COEP headers")
    server.serve_forever()
