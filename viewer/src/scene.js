import * as THREE from 'three';
import { cameraPose, cloudPositions, displayName, distance, vector } from './state.js';
import { calibrate, G8_LENS, validLens } from './calibration.js';

const colors = { yellow: 0xf4d06c, green: 0x91dc83, red: 0xff8877, purple: 0xd89fec, blue: 0x8cbafa };
const hidden = o => o.state !== 'VISIBLE';
// The memory map is drawn from above with an orthographic camera, where point sizes are pixels, not metres.
const MAP_PX = { environment: 2.6, live: 1.6, occluder: 2.6, object: 3.6, ball: 4.6 };

// X-ray blue-cyan by height above the table: the surface dark, anything standing on it bright. Bluer than the
// objects' teal and yellow, so they stand out on it.
function xrayColors(points, tableZ) {
  const out = new Float32Array(points.length * 3);
  points.forEach(([, , z], i) => {
    const t = Math.min(1, Math.max(0, (z - tableZ + 0.03) / 0.25));
    out.set([0.05 + 0.3 * t, 0.16 + 0.55 * t, 0.3 + 0.65 * t], i * 3);
  });
  return out;
}

export class RobotScene {
  constructor(container, labels) {
    this.container = container;
    this.labels = labels;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x101010);
    this.camera = new THREE.PerspectiveCamera(66, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.perspective = this.camera;
    this.mapCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, .01, 100);
    this.view = 'robot';
    this.lens = G8_LENS;
    this.environment = new THREE.Group();
    this.scene.add(this.environment);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    container.append(this.renderer.domElement);
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.healGroup = new THREE.Group();               // the scan's holes, filled in from live depth
    this.scene.add(this.healGroup);
    this.heal = null;
    this.labelItems = [];
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this.render());
  }

  resize() {
    this.dirty = true;
    const { width, height } = this.container.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.frameCamera(width, height);
  }

  frameCamera(width = this.container.clientWidth, height = this.container.clientHeight) {
    if (!width || !height) return;
    this.camera = this.view === 'map' ? this.mapCamera : this.perspective;
    this.viewport = { x: 0, y: 0, width, height };
    if (this.view === 'map') {
      const bounds = this.mapBounds || new THREE.Box3(new THREE.Vector3(-1, -1, 0), new THREE.Vector3(1, 1, 1));
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      const halfH = Math.max(size.y / 2, size.x / 2 * height / width, .4) * 1.5;
      Object.assign(this.camera, { left: -halfH * width / height, right: halfH * width / height, top: halfH, bottom: -halfH });
      this.camera.position.set(center.x, center.y, Math.max(bounds.max.z, 2) + 4);
      this.camera.up.set(0, 1, 0);
      this.camera.lookAt(center.x, center.y, 0);
      this.camera.updateProjectionMatrix();
      this.camera.updateMatrixWorld();
    } else {
      calibrate(this.camera, this.lens, this.lastState?.camera?.T_world_cam);
      const scale = Math.min(width / this.lens.width, height / this.lens.height);
      const w = this.lens.width * scale, h = this.lens.height * scale;
      this.viewport = { x: (width - w) / 2, y: (height - h) / 2, width: w, height: h };
    }
  }

  setView(view) {
    this.view = view;
    if (this.lastState) this.update(this.lastState, this.lastTarget);
    this.resize();
  }

  setScene(data) {
    this.environment.traverse(node => { node.geometry?.dispose(); node.material?.dispose(); });
    this.environment.clear();
    this.staticScene = data;
    this.mapBounds = null;
    this.lens = validLens(data?.camera) ? data.camera : G8_LENS;
    this.envRobot = this.envMap = null;
    if (data) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.cloud.points.flat(), 3));
      this.envRobot = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0x929292,
        size: .008, transparent: true, opacity: .48, depthWrite: false }));
      const mapGeometry = geometry.clone();
      mapGeometry.setAttribute('color', new THREE.Float32BufferAttribute(xrayColors(data.cloud.points, data.table.z), 3));
      this.envMap = new THREE.Points(mapGeometry, new THREE.PointsMaterial({ vertexColors: true, size: MAP_PX.environment,
        sizeAttenuation: false, transparent: true, opacity: .85, depthWrite: false }));
      this.environment.add(this.envRobot, this.envMap);
      const outline = data.table.outline.map(p => new THREE.Vector3(...p, data.table.z));
      if (outline.length >= 3) this.environment.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(outline),
        new THREE.LineBasicMaterial({ color: 0xaaaaaa, transparent: true, opacity: .4 })));
      // The map frames the objects (and the robot, added on the next update), not the whole scan.
      // (no objects in the scene yet: the next update frames the state's objects, e.g. Qwen-3D's answers)
      if (data.objects.length) this.mapBounds = new THREE.Box3().setFromPoints(data.objects.map(o => new THREE.Vector3(...o.position)));
    }
    if (this.lastState) this.update(this.lastState, this.lastTarget);
    this.resize();
  }

  // Points where objects stood, seen bare since: each arrives bright and settles into the scan's colour.
  setHeal(points, ages = []) {
    this.healGroup.traverse(node => { node.geometry?.dispose(); node.material?.dispose(); });
    this.healGroup.clear();
    this.heal = null;
    if (!points.length) return;
    const now = performance.now() / 1000;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points.flat(), 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(points.length * 3), 3));
    const robot = new THREE.Points(geometry, new THREE.PointsMaterial({ vertexColors: true, size: .008,
      transparent: true, opacity: .7, depthWrite: false }));
    const map = new THREE.Points(geometry, new THREE.PointsMaterial({ vertexColors: true, size: MAP_PX.environment + .6,
      sizeAttenuation: false, transparent: true, opacity: .9, depthWrite: false }));
    this.healGroup.add(robot, map);
    const tableZ = this.staticScene?.table?.z ?? 0;
    this.heal = { robot, map, born: points.map((_, i) => now - (Number.isFinite(ages[i]) ? ages[i] : 9)),
      base: { robot: [0.57, 0.57, 0.57], map: Array.from(xrayColors([[0, 0, tableZ]], tableZ)) } };
    this.tintHeal(true);
  }

  tintHeal(force = false) {
    const h = this.heal;
    if (!h) return;
    const now = performance.now() / 1000;
    if (!force && now - Math.max(...h.born) > 3) return;          // all settled: nothing to redraw
    const map = this.view === 'map';
    h.robot.visible = !map; h.map.visible = map;
    const base = map ? h.base.map : h.base.robot, fresh = [0.85, 1, 1];
    const colors = h.robot.geometry.attributes.color;
    h.born.forEach((b, i) => {
      const k = Math.exp(-(now - b) / 0.8);                       // bright when it arrives, settled after ~2 s
      colors.setXYZ(i, ...base.map((c, j) => c + (fresh[j] - c) * k));
    });
    colors.needsUpdate = true;
    this.dirty = true;
  }

  clear() {
    this.dirty = true;
    this.content.traverse(node => {
      node.geometry?.dispose();
      if (Array.isArray(node.material)) node.material.forEach(m => m.dispose());
      else node.material?.dispose();
    });
    this.content.clear();
    this.labels.replaceChildren();
    this.labelItems = [];
  }

  points(positions, color, size, xray = false, opacity = 1, mapPx = MAP_PX.object) {
    if (!positions.length) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const map = this.view === 'map';
    const material = new THREE.PointsMaterial({ color, size: map ? mapPx : size, transparent: true, opacity,
      depthTest: !xray, depthWrite: !xray, sizeAttenuation: !map });
    const points = new THREE.Points(geometry, material);
    points.renderOrder = xray ? 3 : 0;
    this.content.add(points);
  }

  box(object, color, solid = false) {
    const size = vector(object.size) ? object.size : [0.08, 0.08, 0.08];
    const geometry = new THREE.BoxGeometry(...size.map(n => Math.max(0.005, n)));
    if (solid) {
      const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, depthWrite: false }));
      mesh.position.fromArray(object.position);
      this.content.add(mesh);
    }
    const edges = new THREE.EdgesGeometry(geometry);
    if (!solid) geometry.dispose();
    const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color, transparent: true, opacity: solid ? 0.3 : 0.7, depthTest: solid, depthWrite: false }));
    line.renderOrder = 3;
    line.position.fromArray(object.position);
    this.content.add(line);
  }

  label(object, selected) {
    const el = document.createElement('div');
    el.className = `object-label${hidden(object) ? ' hidden-object' : ''}${selected ? ' selected' : ''}`;
    const title = document.createElement('span');
    title.textContent = displayName(object);
    const detail = document.createElement('small');
    const meters = distance(this.pose.position, object.position).toFixed(2);
    detail.textContent = `${meters} m / ${object.state === 'INSIDE' ? `inside cup ${object.inside}` : String(object.state || 'unknown').toLowerCase().replaceAll('_', ' ')}`;
    el.append(title, detail);
    this.labels.append(el);
    const position = new THREE.Vector3(...object.position);
    position.z += (object.size?.[2] || 0.065) * 0.7;
    this.labelItems.push({ el, title, position, name: displayName(object), selected });
  }

  update(state, target) {
    this.lastState = state; this.lastTarget = target;
    const pose = cameraPose(state.camera);
    this.clear();
    if (!pose) return { ready: false, clouds: false };
    this.pose = pose;
    this.camera = this.perspective;
    this.camera.position.fromArray(pose.position);
    const look = new THREE.Vector3(...pose.forward);
    // OpenCV y points down. Preserve measured camera roll when a consistent
    // transform is available; use world-up for legacy forward-only producers.
    this.camera.up.set(0, 0, 1);
    const t = state.camera.T_world_cam;
    if (Array.isArray(t) && t.length === 12 && t.every(Number.isFinite)) {
      const z = new THREE.Vector3(t[2], t[6], t[10]).normalize();
      const up = new THREE.Vector3(-t[1], -t[5], -t[9]);
      if (z.dot(look) > 0.99 && up.length() > 0.9) this.camera.up.copy(up.normalize());
    }
    this.camera.lookAt(this.camera.position.clone().add(look));
    this.camera.updateMatrixWorld();
    this.lens = validLens(state.camera.intrinsics) ? state.camera.intrinsics
      : validLens(this.staticScene?.camera) ? this.staticScene.camera : G8_LENS;
    if (!this.mapBounds) this.mapBounds = new THREE.Box3().setFromPoints(
      [...state.objects, ...state.balls, { position: pose.position }].map(o => new THREE.Vector3(...o.position)));
    else if (!this.mapBounds.robot) {                   // the robot stays in the frame too
      this.mapBounds.expandByPoint(new THREE.Vector3(...pose.position));
      this.mapBounds.robot = true;
    }
    if (this.envRobot) { this.envRobot.visible = this.view !== 'map'; this.envMap.visible = this.view === 'map'; }
    this.tintHeal(true);
    this.frameCamera();

    const floor = Number.isFinite(state.scene?.floor_z) ? state.scene.floor_z : 0;
    const grid = new THREE.GridHelper(8, 40, 0x33463f, 0x27332f);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(0, 0, floor - 0.003);
    grid.material.transparent = true;
    grid.material.opacity = 0.38;
    this.content.add(grid);
    this.points(cloudPositions(state.scene?.point_cloud), 0x637b71, 0.003, false, 0.45, MAP_PX.live);

    for (const occluder of state.occluders) {
      this.box(occluder, 0x9da99f, true);
      this.points(cloudPositions(occluder.point_cloud, occluder.position), 0x9da99f, 0.0028, false, 0.5, MAP_PX.occluder);
    }
    let clouds = !!this.staticScene?.cloud.points.length;
    for (const object of [...state.objects, ...state.balls]) {
      const selected = target?.id === object.id;
      const color = colors[object.color] ?? (hidden(object) ? 0xf4d06c : 0x73e6d3);
      const model = this.staticScene?.objects.find(o => o.id === object.id);
      const points = cloudPositions(model?.points.length ? { frame: 'object', points: model.points } : object.point_cloud, object.position, 2000);
      if (points.length) {
        clouds = true;
        this.points(points, color, object.color ? 0.0025 : 0.0019, true, hidden(object) ? 0.9 : 0.75,
          object.color ? MAP_PX.ball : MAP_PX.object);
      } else this.box(object, color);
      this.label(object, selected);
    }
    if (target) {
      const origin = new THREE.Vector3(...pose.position).addScaledVector(look, 0.17);
      const side = new THREE.Vector3().crossVectors(look, this.camera.up).normalize();
      origin.addScaledVector(new THREE.Vector3().crossVectors(side, look).normalize(), -0.105);
      if (this.view === 'map') origin.fromArray(pose.position);
      const dest = new THREE.Vector3(...target.position);
      const direction = dest.clone().sub(origin);
      if (direction.length() > 0.02) {
        const arrow = new THREE.ArrowHelper(direction.clone().normalize(), origin, direction.length(), 0xf4d06c, 0.025, 0.014);
        arrow.line.material.transparent = true;
        arrow.line.material.opacity = 0.65;
        arrow.line.material.depthTest = false;
        arrow.cone.material.depthTest = false;
        arrow.renderOrder = 4;
        this.content.add(arrow);
      }
    }
    if (this.view === 'map') {
      const origin = new THREE.Vector3(...pose.position);
      const heading = new THREE.Vector3(pose.forward[0], pose.forward[1], 0);
      const marker = new THREE.Mesh(new THREE.SphereGeometry(.025, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
      marker.position.copy(origin); marker.renderOrder = 5; this.content.add(marker);
      if (heading.lengthSq() > .0001) this.content.add(new THREE.ArrowHelper(heading.normalize(), origin, .18, 0xffffff, .04, .035));
    }
    return { ready: true, clouds };
  }

  render() {
    if (!this.dirty) return;
    this.dirty = false;
    const vp = this.viewport;
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, this.container.clientWidth, this.container.clientHeight);
    this.renderer.clear();
    this.renderer.setViewport(vp.x, vp.y, vp.width, vp.height);
    this.renderer.setScissor(vp.x, vp.y, vp.width, vp.height);
    this.renderer.setScissorTest(true);
    this.renderer.render(this.scene, this.camera);
    const width = this.container.clientWidth, height = this.container.clientHeight;
    const placed = [];
    for (const item of [...this.labelItems].sort((a, b) => Number(b.selected) - Number(a.selected))) {
      const cameraSpace = item.position.clone().applyMatrix4(this.camera.matrixWorldInverse);
      const projected = item.position.clone().project(this.camera);
      const offscreen = cameraSpace.z >= 0 || Math.abs(projected.x) > 0.9 || Math.abs(projected.y) > 0.8;
      let px = projected.x, py = projected.y;
      if (offscreen) {
        // Camera-space bearing remains well-defined for points behind the robot.
        px = cameraSpace.x;
        py = cameraSpace.y;
        if (Math.abs(px) + Math.abs(py) < 1e-6) py = -1;
        const scale = Math.max(Math.abs(px) / 0.72, Math.abs(py) / 0.65);
        px /= scale; py /= scale;
      }
      item.el.classList.toggle('edge', offscreen);
      const bearing = Math.abs(cameraSpace.x) > Math.abs(cameraSpace.y)
        ? (cameraSpace.x < 0 ? '\u2190 ' : '\u2192 ')
        : (cameraSpace.y > 0 ? '\u2191 ' : '\u2193 ');
      item.title.textContent = `${offscreen ? bearing : ''}${item.name}`;
      const preferredX = vp.x + (px * .5 + .5) * vp.width, preferredY = vp.y + (-py * .5 + .5) * vp.height;
      const labelWidth = item.el.offsetWidth, labelHeight = item.el.offsetHeight;
      let spot;
      for (const dy of [0, -44, 44, -88, 88, -132, 132]) {
        for (const dx of [0, -120, 120, -60, 60]) {
          const x = THREE.MathUtils.clamp(preferredX + dx, labelWidth / 2 + 12, width - labelWidth / 2 - 12);
          const y = THREE.MathUtils.clamp(preferredY + dy, 58 + labelHeight, height - 45);
          const rect = { left: x - labelWidth / 2, right: x + labelWidth / 2, top: y - labelHeight, bottom: y };
          if (placed.every(p => rect.right + 7 < p.left || rect.left - 7 > p.right || rect.bottom + 7 < p.top || rect.top - 7 > p.bottom)) {
            spot = { x, y, rect }; break;
          }
        }
        if (spot) break;
      }
      item.el.style.visibility = spot ? 'visible' : 'hidden';
      if (spot) {
        placed.push(spot.rect);
        item.el.style.left = `${spot.x}px`; item.el.style.top = `${spot.y}px`;
      }
    }
  }
}
