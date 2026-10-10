import json
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen

from core.jetson.metrics_server import Metrics, handler
from http.server import ThreadingHTTPServer


class MetricsTests(unittest.TestCase):
    def test_non_jetson_never_reports_host_memory(self):
        with patch('core.jetson.metrics_server.device_model', return_value=None), \
             patch('core.jetson.metrics_server.Sampler') as sampler:
            monitor = Metrics()
            sample = monitor.latest()
            self.assertFalse(sample['available'])
            self.assertEqual(sample['metrics'], {})
            sampler.assert_not_called()
            monitor.stop()

    def test_live_partial_and_stale_samples(self):
        sampler = Mock(t0=100)
        sample = dict(t=1, ram_used_gb=4, ram_total_gb=8, gpu_pct=65, cpu_pct=20,
                      power_w=9.2, temp_c=52, swap_used_mb=0)
        sampler.latest.return_value = sample
        with patch('core.jetson.metrics_server.device_model', return_value='NVIDIA Jetson Orin Nano'), \
             patch('core.jetson.metrics_server.Sampler', return_value=sampler), \
             patch('core.jetson.metrics_server.time.monotonic', return_value=101.2):
            monitor = Metrics()
            self.assertEqual(monitor.latest()['reason'], 'live')
            self.assertEqual(monitor.latest()['age_ms'], 200)
            self.assertEqual(monitor.latest()['memory_kind'], 'shared')
            del sample['gpu_pct']
            self.assertEqual(monitor.latest()['reason'], 'partial')
            self.assertIsNone(monitor.latest()['metrics']['gpu_pct'])
            sample['t'] = -10
            self.assertFalse(monitor.latest()['available'])
            self.assertEqual(monitor.latest()['reason'], 'stale')
            monitor.stop()
            sampler.stop.assert_called_once()

    def test_http_contract_and_cors(self):
        with patch('core.jetson.metrics_server.device_model', return_value=None):
            monitor = Metrics()
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(monitor))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f'http://127.0.0.1:{server.server_port}/metrics') as response:
                self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertEqual(response.headers['Content-Type'], 'application/json')
                self.assertEqual(json.load(response)['reason'], 'not_jetson')
        finally:
            server.shutdown(); server.server_close(); thread.join()
