export type Crop = { x: number; y: number; w: number; h: number };
export type Source = HTMLVideoElement | HTMLImageElement;

export function dimensions(source: Source | null): [number, number] {
  if (!source) return [0, 0];
  return source instanceof HTMLVideoElement
    ? [source.videoWidth, source.videoHeight] : [source.naturalWidth, source.naturalHeight];
}

export function frameVersion(source: Source): number {
  if (!(source instanceof HTMLVideoElement)) return 0;
  return source.getVideoPlaybackQuality?.().totalVideoFrames || source.currentTime;
}

export function centered(width: number, height: number, aspect: number): Crop {
  const w = Math.min(1, height * aspect / width);
  const h = Math.min(1, width / aspect / height);
  return { x: (1 - w) / 2, y: (1 - h) / 2, w, h };
}

function pixelCrop(source: Source, crop?: Crop) {
  const [width, height] = dimensions(source);
  const c = crop || { x: 0, y: 0, w: 1, h: 1 };
  const x = Math.floor(c.x * width + .5), y = Math.floor(c.y * height + .5);
  const w = Math.min(width - x, Math.floor(c.w * width + .5));
  const h = Math.min(height - y, Math.floor(c.h * height + .5));
  return { width, x, y, w, h };
}

export function drawFrame(ctx: CanvasRenderingContext2D, source: Source, mirrored = false, crop?: Crop) {
  const { width, x, y, w, h } = pixelCrop(source, crop);
  ctx.save();
  if (mirrored) { ctx.translate(ctx.canvas.width, 0); ctx.scale(-1, 1); }
  ctx.drawImage(source, mirrored ? width - x - w : x, y, w, h, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();
}

export function jpeg(source: Source, crop?: Crop, mirrored = false, canvas?: HTMLCanvasElement): Promise<Blob> {
  const { w, h } = pixelCrop(source, crop);
  if (!w || !h) return Promise.reject(new Error('Camera frame is not ready.'));
  const scale = crop ? Math.min(1, 480 / h) : 1;
  const target = canvas ?? document.createElement('canvas');
  const width = Math.max(1, Math.round(w * scale)), height = Math.max(1, Math.round(h * scale));
  if (target.width !== width || target.height !== height) { target.width = width; target.height = height; }
  drawFrame(target.getContext('2d')!, source, mirrored, crop);
  return new Promise((resolve, reject) => target.toBlob(
    blob => blob ? resolve(blob) : reject(new Error('Image encoding failed.')), 'image/jpeg', .9));
}

export function packet(header: object, payload: Blob): Blob {
  const data = new TextEncoder().encode(JSON.stringify(header));
  const size = new ArrayBuffer(4);
  new DataView(size).setUint32(0, data.length);
  return new Blob([size, data, payload]);
}

export function unpack(data: ArrayBuffer): { header: any; payload: Blob } {
  const size = new DataView(data).getUint32(0);
  if (size > 8192 || size + 4 >= data.byteLength) throw new Error('Invalid output frame.');
  return { header: JSON.parse(new TextDecoder().decode(data.slice(4, size + 4))),
    payload: new Blob([data.slice(size + 4)], { type: 'image/jpeg' }) };
}
