"""Timestamp-matched camera playback for a private xray.frames folder."""
from bisect import bisect_left, bisect_right
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from xray.frames import frame_ids, project


class RecordedTrack:
    def __init__(self, folder, states):
        self.folder = Path(folder)
        self.ids = frame_ids(self.folder)
        meta = json.loads((self.folder / 'meta.json').read_text())
        self.times = np.asarray(meta['frame_times_s'], dtype=float)
        self.state_times = np.asarray([s['t'] for s in states], dtype=float)
        for times in (self.times, self.state_times):
            if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError('Recording timestamps must be finite and strictly increasing')
        if len(self.ids) != len(self.times):
            raise ValueError('Track images and metadata frame count differ')
        self.matches = []
        for t in self.state_times:
            i = bisect_left(self.times, t)
            candidates = [j for j in (i - 1, i) if 0 <= j < len(self.times)]
            match = min(candidates, key=lambda j: abs(self.times[j] - t))
            if abs(self.times[match] - t) > .05:
                raise ValueError(f'No camera frame within 50 ms of state t={t}')
            self.matches.append(match)
        self.calibrations = {}
        for i in set(self.matches):
            k = self.ids[i]
            K = np.loadtxt(self.folder / 'intrinsic' / f'{k}.txt').reshape(3, 3)
            T = np.loadtxt(self.folder / 'pose' / f'{k}.txt').reshape(4, 4)
            if not np.isfinite(K).all() or not np.isfinite(T).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
                raise ValueError(f'Invalid calibration for frame {k}')
            self.calibrations[i] = (K, T)
        self.width, self.height = meta['image_size_wh']
        step = float(np.median(np.diff(self.state_times))) if len(states) > 1 else 1 / 15
        self.duration = float(self.state_times[-1] - self.state_times[0] + step)

    def state_index(self, elapsed):
        t = self.state_times[0] + max(0, elapsed) % self.duration
        return max(0, bisect_right(self.state_times, t) - 1)

    def enrich(self, state, index):
        i = self.matches[index]
        K, T = self.calibrations[i]
        camera = dict(state.get('camera', {}), position=T[:3, 3].tolist(),
                      forward=T[:3, 2].tolist(), T_world_cam=T[:3].reshape(-1).tolist(),
                      intrinsics=dict(width=self.width, height=self.height, fx=K[0, 0],
                                      fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]))
        return dict(state, camera=camera, replay=dict(index=index, frame=self.ids[i],
                    camera=True, frame_t=float(self.times[i])))

    def image(self, state, index, raw=False):
        i = self.matches[index]
        body = (self.folder / 'color' / f'{self.ids[i]}.jpg').read_bytes()
        if raw:
            return body
        image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Invalid recorded JPEG')
        K, T = self.calibrations[i]
        balls = state.get('ball', {}).get('balls') or ([state['ball']] if state.get('ball') else [])
        objects = state.get('objects', []) + balls
        target = next((o for o in objects if o.get('id') == state.get('target_id')), None)
        if state.get('target_id') is None:
            target = next((o for o in balls if o.get('color') == 'yellow'), None)
        labels = []
        for o in objects:
            u, v, z = project(np.asarray([o['position']]), K, T)
            if z[0] <= .01 or not (0 <= u[0] < image.shape[1] and 0 <= v[0] < image.shape[0]):
                continue
            center = (int(round(u[0])), int(round(v[0])))
            color = (108, 208, 244) if o.get('state') != 'VISIBLE' or o is target else (211, 230, 115)
            cv2.circle(image, center, 7, color, 2, cv2.LINE_AA)
            name = f"{o['color']} ball" if o.get('color') else f"{o.get('label', 'object')} {o['id']}"
            label = f"{name} | {o.get('state', '')}"
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .35, 1)
            for dy in (-14, 24, -38, 48, -62, 72):
                x = max(4, min(center[0] + 10, image.shape[1] - w - 8))
                y = max(40, min(center[1] + dy, image.shape[0] - 30))
                rect = (x - 3, y - h - 3, x + w + 3, y + baseline + 3)
                if any(rect[0] < r[2] + 3 and rect[2] > r[0] - 3 and rect[1] < r[3] + 3 and rect[3] > r[1] - 3 for r in labels):
                    continue
                labels.append(rect)
                cv2.rectangle(image, rect[:2], rect[2:], (18, 24, 22), -1)
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
                break
            if o is target:
                cv2.arrowedLine(image, (image.shape[1] // 2, image.shape[0] - 20), center, color, 1, cv2.LINE_AA, tipLength=.04)
        ok, jpg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError('Could not encode replay overlay')
        return jpg.tobytes()
