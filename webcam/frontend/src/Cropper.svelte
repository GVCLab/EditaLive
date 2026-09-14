<script lang="ts">
  import { onMount } from 'svelte';
  import { centered, dimensions, drawFrame, frameVersion, type Crop, type Source } from './media';
  export let source: Source | null = null;
  export let aspect = 1.75;
  export let previewAspect = 1.75;
  export let crop: Crop = { x: 0, y: 0, w: 1, h: 1 };
  export let disabled = false;
  export let label = 'Frame';
  export let mirrored = false;
  let canvas: HTMLCanvasElement, preview: HTMLDivElement, frame: HTMLDivElement;
  let previewWidth = 0, previewHeight = 0;
  let sizeKey = '', width = 0, height = 0;
  type Corner = 'nw' | 'ne' | 'sw' | 'se';
  const corners: { id: Corner; name: string; right: boolean; bottom: boolean }[] = [
    { id:'nw', name:'top left', right:false, bottom:false },
    { id:'ne', name:'top right', right:true, bottom:false },
    { id:'sw', name:'bottom left', right:false, bottom:true },
    { id:'se', name:'bottom right', right:true, bottom:true },
  ];
  let drag: { x: number; y: number; crop: Crop; corner?: Corner; pointer: number } | null = null;
  let raf = 0;
  let lastSource: Source | null = null, drawKey = '';
  const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

  function home() {
    if (width && height) crop = centered(width, height, aspect);
  }
  function translate(start: Crop, dx: number, dy: number) {
    const rect = frame.getBoundingClientRect();
    crop = { ...start, x:clamp(start.x + dx / rect.width, 0, 1 - start.w),
      y:clamp(start.y + dy / rect.height, 0, 1 - start.h) };
  }
  function resize(start: Crop, corner: Corner, dx: number, dy: number) {
    const rect = frame.getBoundingClientRect();
    const sx = corner.endsWith('e') ? 1 : -1, sy = corner.startsWith('s') ? 1 : -1;
    const ax = start.x + (sx < 0 ? start.w : 0), ay = start.y + (sy < 0 ? start.h : 0);
    const ratio = aspect * height / width;
    const delta = (sx * dx + sy * dy / aspect) / (1 + 1 / aspect ** 2);
    const max = Math.min(sx > 0 ? 1 - ax : ax, (sy > 0 ? 1 - ay : ay) * ratio);
    const w = clamp(start.w + delta / rect.width, centered(width, height, aspect).w / 4, max);
    const h = w / ratio;
    crop = { x:ax - (sx < 0 ? w : 0), y:ay - (sy < 0 ? h : 0), w, h };
  }
  function down(event: PointerEvent, corner?: Corner) {
    if (disabled || !width || event.button !== 0) return;
    event.preventDefault();
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    drag = { x:event.clientX, y:event.clientY, crop:{...crop}, corner, pointer:event.pointerId };
  }
  function move(event: PointerEvent) {
    if (!drag || disabled || event.pointerId !== drag.pointer) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (drag.corner) resize(drag.crop, drag.corner, dx, dy);
    else translate(drag.crop, dx, dy);
  }
  function zoom(factor: number) {
    const base = centered(width, height, aspect);
    const w = clamp(crop.w * factor, base.w / 4, base.w), h = w * width / height / aspect;
    crop = { x:clamp(crop.x + (crop.w - w) / 2, 0, 1 - w),
      y:clamp(crop.y + (crop.h - h) / 2, 0, 1 - h), w, h };
  }
  function wheel(event: WheelEvent) {
    if (disabled || !width) return;
    event.preventDefault();
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? previewHeight : 1);
    zoom(Math.exp(clamp(delta * .002, -.3, .3)));
  }
  function key(event: KeyboardEvent, corner?: Corner) {
    if (disabled || !width) return;
    const amount = event.shiftKey ? 10 : 2;
    const dx = event.key === 'ArrowLeft' ? -amount : event.key === 'ArrowRight' ? amount : 0;
    const dy = event.key === 'ArrowUp' ? -amount : event.key === 'ArrowDown' ? amount : 0;
    if (dx || dy) {
      if (corner) resize(crop, corner, dx, dy); else translate(crop, dx, dy);
    } else if (event.key === 'Home') home();
    else if (event.key === '+' || event.key === '=') zoom(.95);
    else if (event.key === '-') zoom(1.05);
    else return;
    event.preventDefault();
  }
  function draw() {
    raf = requestAnimationFrame(draw);
    [width, height] = dimensions(source);
    if (width && height && source) {
      const key = `${width}:${height}:${aspect}`;
      if (key !== sizeKey && !disabled) { sizeKey = key; home(); }
      const w = Math.min(800, width), h = Math.round(w * height / width);
      const nextKey = `${w}:${h}:${mirrored}:${crop.x}:${crop.y}:${crop.w}:${crop.h}:${frameVersion(source)}`;
      if (source === lastSource && nextKey === drawKey) return;
      lastSource = source; drawKey = nextKey;
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      const ctx = canvas.getContext('2d')!;
      drawFrame(ctx, source, mirrored);
      const x = crop.x * w, y = crop.y * h, cw = crop.w * w, ch = crop.h * h;
      ctx.fillStyle = '#07111ba8';
      ctx.fillRect(0, 0, w, y); ctx.fillRect(0, y + ch, w, h - y - ch);
      ctx.fillRect(0, y, x, ch); ctx.fillRect(x + cw, y, w - x - cw, ch);
      ctx.strokeStyle = '#ffffff40'; ctx.lineWidth = 1;
      for (const n of [1, 2]) {
        ctx.beginPath(); ctx.moveTo(x + cw * n / 3, y); ctx.lineTo(x + cw * n / 3, y + ch);
        ctx.moveTo(x, y + ch * n / 3); ctx.lineTo(x + cw, y + ch * n / 3); ctx.stroke();
      }
    }
  }
  onMount(() => {
    const observer = new ResizeObserver(([entry]) => {
      previewWidth = entry.contentRect.width; previewHeight = entry.contentRect.height;
    });
    observer.observe(preview); draw();
    return () => { cancelAnimationFrame(raf); observer.disconnect(); };
  });
</script>

<div class="cropper">
  <div bind:this={preview} class="preview" class:has-source={!!width} style={`aspect-ratio:${previewAspect}`}>
    <div bind:this={frame} class="frame" style={`width:${width && height ? Math.min(previewWidth, previewHeight * width / height) : 0}px;aspect-ratio:${width && height ? width / height : 1}`}>
      <canvas bind:this={canvas} class:empty={!width} aria-label={`${label} crop preview`} />
      {#if width}
        <button class="crop-area" type="button" {disabled} aria-label={`Move ${label.toLowerCase()} crop`}
          title="Drag to move · Scroll to zoom · Double-click to reset"
          style={`left:${crop.x * 100}%;top:${crop.y * 100}%;width:${crop.w * 100}%;height:${crop.h * 100}%`}
          on:pointerdown={e => down(e)} on:pointermove={move} on:pointerup={() => drag = null}
          on:pointercancel={() => drag = null} on:lostpointercapture={() => drag = null}
          on:wheel|nonpassive={wheel} on:keydown={e => key(e)} on:dblclick={home}></button>
        {#each corners as corner}
          <button class={`handle ${corner.id}`} type="button" {disabled}
            aria-label={`Resize ${label.toLowerCase()} crop ${corner.name}`} title="Drag to resize"
            style={`left:clamp(10px,${(crop.x + (corner.right ? crop.w : 0)) * 100}%,calc(100% - 10px));top:clamp(10px,${(crop.y + (corner.bottom ? crop.h : 0)) * 100}%,calc(100% - 10px))`}
            on:pointerdown={e => down(e, corner.id)} on:pointermove={move} on:pointerup={() => drag = null}
            on:pointercancel={() => drag = null} on:lostpointercapture={() => drag = null}
            on:keydown={e => key(e, corner.id)}></button>
        {/each}
      {/if}
    </div>
    {#if !width}<slot><span>Waiting for {label.toLowerCase()}</span></slot>{/if}
  </div>
</div>

<style>
  .cropper { min-width:0; }
  .preview { position:relative; width:100%; background-color:#f3ecee; background-image:linear-gradient(#a7839307 1px,transparent 1px),linear-gradient(90deg,#a7839307 1px,transparent 1px); background-size:28px 28px; border:1px solid #e3d7dc; border-radius:13px; overflow:hidden; display:grid; place-items:center; }
  .preview.has-source { background:#171723; }
  .frame { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); }
  canvas { display:block; width:100%; height:100%; pointer-events:none; } canvas.empty { visibility:hidden; }
  .preview>span { color:#a28da7; font-size:12px; }
  button { position:absolute; margin:0; padding:0; touch-action:none; background:transparent; }
  .crop-area { border:1.5px solid #8ee2c6; cursor:move; }
  .crop-area:disabled { border-color:#e3e8ed; cursor:default; }
  .handle { width:24px; height:24px; transform:translate(-50%,-50%); border:0; display:grid; place-items:center; }
  .handle::after { content:''; width:9px; height:9px; border:1.5px solid #62ac94; border-radius:3px; background:#fffdf8; box-shadow:0 1px 4px #0004; }
  .nw,.se { cursor:nwse-resize; } .ne,.sw { cursor:nesw-resize; }
  .handle:disabled { visibility:hidden; }
</style>
