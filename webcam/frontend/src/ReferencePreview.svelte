<script lang="ts">
  import { onMount } from 'svelte';
  import Icon from './Icon.svelte';
  import { dimensions, drawFrame, frameVersion, type Crop, type Source } from './media';
  export let source: Source | null = null;
  export let aspect = 1.75;
  export let crop: Crop;
  export let mirrored = false;
  let canvas: HTMLCanvasElement;
  let ready = false, raf = 0;
  let lastSource: Source | null = null, drawKey = '';

  function draw() {
    raf = requestAnimationFrame(draw);
    const [width, height] = dimensions(source);
    ready = !!(width && height);
    if (ready && source) {
      const w = 320, h = Math.round(w / aspect);
      const nextKey = `${w}:${h}:${mirrored}:${crop.x}:${crop.y}:${crop.w}:${crop.h}:${frameVersion(source)}`;
      if (source === lastSource && nextKey === drawKey) return;
      lastSource = source; drawKey = nextKey;
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      drawFrame(canvas.getContext('2d')!, source, mirrored, crop);
    }
  }
  onMount(() => { draw(); return () => cancelAnimationFrame(raf); });
</script>

<div class="reference-preview">
  <canvas bind:this={canvas} class:empty={!ready} aria-label="Reference preview" />
  {#if !ready}<div class="placeholder"><Icon name="image" size={25} /><span>From your camera</span></div>{/if}
</div>

<style>
  .reference-preview { flex:1 0 auto; min-height:0; position:relative; width:100%; aspect-ratio:7 / 4; overflow:hidden; border:1px solid #e3d7dc; border-radius:11px; background:#f3ecee; display:grid; place-items:center; }
  canvas { position:absolute; width:100%; height:100%; object-fit:contain; } canvas.empty { visibility:hidden; }
  .placeholder { display:grid; justify-items:center; gap:8px; color:#8d7995; font-size:10px; padding:10px; }
</style>
