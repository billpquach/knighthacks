"""Read-only Jetson telemetry: python -m core.jetson.metrics_server --host 0.0.0.0.

GET /metrics serves one current sample. No dependencies beyond the standard
library; tegrastats must be on PATH for GPU/CPU/temperature/power readings.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time

from .jetson_stats import Sampler


def device_model():
    for name in ('/proc/device-tree/model', '/sys/firmware/devicetree/base/model'):
        try:
            model = Path(name).read_text().strip('\x00\n ')
            if 'jetson' in model.lower():
                return model
            compatible = Path(name).with_name('compatible')
            tegra = compatible.exists() and 'nvidia,tegra' in compatible.read_text()
            if tegra:
                return model
        except OSError:
            pass
    return None


class Metrics:
    """One sampler per server, never start tegrastats for individual requests."""
    def __init__(self):
        self.model = device_model()
        self.sampler = Sampler(interval=1, keep=120) if self.model else None

    def latest(self):
        result = dict(version=1, device=self.model, memory_kind='shared', available=False,
                      reason='not_jetson', sample_id=None, age_ms=None, metrics={})
        if self.sampler is None:
            return result
        sample = self.sampler.latest()
        age = max(0, (time.monotonic() - self.sampler.t0 - sample['t']) * 1000)
        fields = ('ram_used_gb', 'ram_total_gb', 'gpu_pct', 'cpu_pct', 'power_w', 'temp_c', 'swap_used_mb')
        values = {key: sample.get(key) for key in fields}
        available = age < 5000 and any(v is not None for v in values.values())
        partial = any(v is None for v in values.values())
        result.update(available=available, reason='stale' if age >= 5000 else 'partial' if partial else 'live',
                      sample_id=f'{self.sampler.t0}:{sample["t"]}', age_ms=round(age), metrics=values)
        return result

    def stop(self):
        if self.sampler:
            self.sampler.stop()


def handler(metrics):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def do_GET(self):
            if self.path.split('?')[0] != '/metrics':
                self.send_error(404)
                return
            body = json.dumps(metrics.latest(), allow_nan=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8083)
    args = parser.parse_args()
    metrics = Metrics()
    server = ThreadingHTTPServer((args.host, args.port), handler(metrics))
    print(f'Metrics: http://{args.host}:{args.port}/metrics', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        metrics.stop()


if __name__ == '__main__':
    main()
