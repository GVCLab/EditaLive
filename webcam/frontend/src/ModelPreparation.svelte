<script lang="ts">
  import { onMount } from 'svelte';
  import Icon from './Icon.svelte';

  export let progress: {
    elapsed_seconds: number;
    workers: Record<string, { message: string; done: boolean }>;
  } | null = null;

  let now = Date.now();
  $: startedAt = Date.now() - (progress?.elapsed_seconds ?? 0) * 1000;
  $: seconds = Math.max(0, Math.floor((now - startedAt) / 1000));
  $: elapsed = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;

  onMount(() => {
    const timer = setInterval(() => now = Date.now(), 1000);
    return () => clearInterval(timer);
  });
</script>

<section class="preparation" aria-label="Model preparation progress">
  <div class="heading"><span>Preparing model</span><span aria-label="Preparation elapsed time">{elapsed} elapsed</span></div>
  <div class="workers" role="status" aria-live="polite">
    {#each [['generation', 'Generation'], ['extraction', 'Motion tracking']] as [key, label]}
      {@const worker = progress?.workers[key]}
      <div class="worker" class:done={worker?.done}>
        <span class="indicator" aria-hidden="true">{#if worker?.done}<Icon name="check" size={12} />{:else}<span class="spinner"></span>{/if}</span>
        <span class="label">{label}</span><span class="message">{worker?.message ?? 'Starting worker'}</span>
      </div>
    {/each}
  </div>
</section>

<style>
  .preparation { margin-top:14px; padding:8px 10px; border:1px solid #e2d8eb; border-radius:10px; background:#f6f0fa; color:#746080; font-size:11px; line-height:1.4; }
  .heading { display:flex; justify-content:space-between; gap:8px; margin-bottom:6px; font-size:10px; color:#887393; }
  .heading span:first-child { font-weight:600; color:#7b5d9a; }
  .heading span:last-child { font-variant-numeric:tabular-nums; white-space:nowrap; }
  .workers { display:grid; gap:5px; }
  .worker { display:grid; grid-template-columns:12px 90px minmax(0,1fr); align-items:start; gap:6px; }
  .indicator { display:flex; align-items:center; height:15px; }
  .label { font-weight:600; }
  .message { overflow-wrap:anywhere; }
  .done { color:#548675; }
  .spinner { width:10px; height:10px; box-sizing:border-box; border:1.5px solid #d8c8e7; border-top-color:#9675b1; border-radius:50%; animation:spin 1s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  @media(min-width:1100px) and (max-height:800px) { .preparation { margin-top:10px; } }
  @media(prefers-reduced-motion:reduce) { .spinner { animation:none; } }
</style>
