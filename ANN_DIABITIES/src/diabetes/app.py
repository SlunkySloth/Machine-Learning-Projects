"""Local web interface: PYTHONPATH=src python -m diabetes.app."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import urlsplit
import joblib
from threadpoolctl import threadpool_limits
from .data import ROOT
from .predict import predict, what_if


def make_handler(bundle):
    inference_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def send_bytes(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/results":
                self.send_json(json.loads((ROOT / "reports/results.json").read_text()))
                return
            if path == "/api/health":
                self.send_json({"status": "ok", "model_loaded": True})
                return
            static = {"/": ROOT / "web/index.html", "/styles.css": ROOT / "web/styles.css",
                      "/app.js": ROOT / "web/app.js", "/presentation": ROOT / "web/presentation.html"}
            file = static.get(path)
            if path.startswith("/figures/"):
                name = path.removeprefix("/figures/")
                if "/" not in name and "\\" not in name and name.endswith(".png"):
                    file = ROOT / "reports/figures" / name
            if file and file.is_file():
                self.send_bytes(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            else:
                self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ["/api/predict", "/api/what-if"]:
                self.send_json({"error": "Not found"}, 404)
                return
            # A local application: reject cross-origin form / browser requests.
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.send_json({"error": "Cross-origin requests are not supported."}, 403)
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_json({"error": "Send application/json."}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Request body must be between 1 and 16384 bytes.")
                payload = json.loads(self.rfile.read(length))
                with inference_lock:
                    if path == "/api/predict":
                        result = predict(bundle, payload)
                    else:
                        if not isinstance(payload, dict) or set(payload) != {"inputs", "feature", "value"}:
                            raise ValueError("What-if requires inputs, feature and value.")
                        result = what_if(bundle, payload["inputs"], payload["feature"], payload["value"])
                self.send_json(result)
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json({"error": str(exc)}, 400)

        def log_message(self, format, *args):
            # Do not persist input measurements or prediction payloads.
            print(f"{self.command} {urlsplit(self.path).path} — {args[1] if len(args) > 1 else ''}")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    path = ROOT / "artifacts/model.joblib"
    if not path.exists():
        parser.error("Train first: PYTHONPATH=src python -m diabetes.train")
    bundle = joblib.load(path)
    with threadpool_limits(limits=1):
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bundle))
        print(f"Diabetes study interface: http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
