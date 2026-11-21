// map3d offline 3D Tiles viewer.
//
// Renders a map3d tileset in its local scene-centre ENU frame. glTF content is
// already y-up as (E, U, -N), so the root ENU->ECEF transform is not applied.
// Tiles refine by screen-space error (REPLACE): a parent stays visible until all
// of its children have loaded, so there are no holes while streaming.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const MAX_SSE_PX = 16;
const status = document.getElementById('status');
const container = document.getElementById('view');

const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdde8f2);
// Conceptual style lighting: a soft
// sky with a light ground bounce keeps shaded walls light grey, and a sun 35
// degrees up in the south-west lights south- and west-facing walls and leaves
// east- and north-facing ones in shade, so roofs, sunlit walls and shaded walls
// read apart from any southern viewpoint.
scene.add(new THREE.HemisphereLight(0xffffff, 0xe0dcd2, 1.7));
const sun = new THREE.DirectionalLight(0xfff8ee, 1.9);
sun.position.set(-0.6, 0.6, 0.6);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 200000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;

const loader = new GLTFLoader();
let loading = 0;

// ENU (z-up) box -> three.js (y-up) Box3.
function boxToThree(box) {
  const [cx, cy, cz, hx, , , , hy, , , , hz] = box;
  return new THREE.Box3(
    new THREE.Vector3(cx - hx, cz - hz, -(cy + hy)),
    new THREE.Vector3(cx + hx, cz + hz, -(cy - hy)),
  );
}

function prepare(tile) {
  tile.box3 = boxToThree(tile.boundingVolume.box);
  tile.object = null;
  tile.state = 'idle';
  for (const child of tile.children || []) prepare(child);
}

function load(tile) {
  if (tile.state !== 'idle' || !tile.content) return;
  tile.state = 'loading';
  loading += 1;
  loader.load(tile.content.uri, (gltf) => {
    tile.object = gltf.scene;
    tile.object.visible = false;
    scene.add(tile.object);
    tile.state = 'ready';
    loading -= 1;
  }, undefined, (err) => {
    console.error('failed to load', tile.content.uri, err);
    tile.state = 'error';
    loading -= 1;
  });
}

function hideSubtree(tile) {
  if (tile.object) tile.object.visible = false;
  for (const child of tile.children || []) hideSubtree(child);
}

function screenSpaceError(tile) {
  const distance = tile.box3.distanceToPoint(camera.position);
  if (distance <= 0) return Infinity;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  return (tile.geometricError * renderer.domElement.clientHeight) / (distance * 2 * Math.tan(fov / 2));
}

function update(tile) {
  load(tile);
  const children = tile.children || [];
  if (children.length && screenSpaceError(tile) > MAX_SSE_PX) {
    children.forEach(load);
    if (children.every((c) => c.state === 'ready' || c.state === 'error')) {
      if (tile.object) tile.object.visible = false;
      children.forEach(update);
      return;
    }
  }
  if (tile.object) tile.object.visible = true;
  children.forEach(hideSubtree);
}

function showAttribution(tileset) {
  const items = (tileset.asset.extras && tileset.asset.extras.attribution) || [];
  const panel = document.getElementById('attribution');
  const list = document.createElement('ul');
  for (const text of items) {
    const li = document.createElement('li');
    li.textContent = text;
    list.appendChild(li);
  }
  panel.replaceChildren(list);
}

const tileset = await (await fetch('tileset.json')).json();
prepare(tileset.root);
showAttribution(tileset);
const extent = tileset.root.box3;
const centre = extent.getCenter(new THREE.Vector3());
const size = extent.getSize(new THREE.Vector3());
const radius = Math.max(size.x, size.z) * 0.6;
controls.target.copy(centre);
camera.position.set(centre.x + radius * 0.6, centre.y + radius * 0.8, centre.z + radius * 0.9);

// Optional view in the URL: #cam=E,N,U,targetE,targetN,targetU (scene ENU metres).
// The hash follows the camera, so a view can be bookmarked or shared.
const enuToThree = (e, n, u) => new THREE.Vector3(e, u, -n);
const fromHash = /cam=([-\d.,e]+)/.exec(window.location.hash);
const parts = fromHash ? fromHash[1].split(',').map(Number) : [];
if (parts.length === 6 && parts.every(Number.isFinite)) {
  camera.position.copy(enuToThree(parts[0], parts[1], parts[2]));
  controls.target.copy(enuToThree(parts[3], parts[4], parts[5]));
}
controls.addEventListener('end', () => {
  const p = camera.position;
  const t = controls.target;
  const view = [p.x, -p.z, p.y, t.x, -t.z, t.y].map((v) => v.toFixed(1)).join(',');
  history.replaceState(null, '', `#cam=${view}`);
});
controls.update();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  controls.update();
  update(tileset.root);
  status.textContent = loading ? `Loading ${loading} tile(s)…` : 'Ready';
  renderer.render(scene, camera);
});
