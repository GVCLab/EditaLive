<script lang="ts">
  import { onMount } from 'svelte';
  import Cropper from './Cropper.svelte';
  import ReferencePreview from './ReferencePreview.svelte';
  import Icon from './Icon.svelte';
  import ModelPreparation from './ModelPreparation.svelte';
  import { PlaybackClock } from './playback';
  import { jpeg, packet, unpack, type Crop, type Source } from './media';

  let resolution = 384, fp8 = true, decoder = 'flash', dualGpu = false;
  let prompt = 'Re-render it in a Pixar-style 3D animation look with smooth shading, soft global illumination, and vibrant colors.';
  let captureTargetFps = 15;
  let preparing: 'model' | 'prompt' | null = null;
  let status: any = { state: 'idle', message: 'Choose your settings and prepare', driving_fps: 15 };
  let connected = false, error = '', busy = false, cameraReady = false;
  let socket: WebSocket | null = null, stream: MediaStream | null = null;
  let camera: HTMLVideoElement, output: HTMLCanvasElement, captureCanvas: HTMLCanvasElement;
  let uploadWidth = 0, uploadHeight = 0;
  let capturedReference: HTMLImageElement | null = null;
  let referenceSource: Source | null = null;
  let frameCrop: Crop = { x: 0, y: 0, w: 1, h: 1 };
  let session: string | null = null, credits = 0, sequence = 0, encoding = false;
  let captureFps = 0, playbackFps = 0, latency = 0, shown = 0;
  let captureCount = 0, playbackCount = 0, statsTime = 0, nextCapture = 0;
  let stopped = false, lastVideoTime = -1, raf = 0;
  let playbackClock = new PlaybackClock();
  let decodeChain = Promise.resolve();
  let playback: { bitmap: ImageBitmap; header: any }[] = [];

  $: config = { resolution, orientation: 'landscape', fp8, decoder, dual_gpu: dualGpu };
  $: width = resolution === 384 ? 672 : 832;
  $: height = resolution;
  $: aspect = width / height;
  $: running = status.state === 'running';
  $: modelPreparing = connected && (preparing === 'model' || !!status.preparation);
  $: if (!busy) preparing = null;
  $: locked = busy || ['preparing', 'starting', 'running', 'stopping'].includes(status.state);
  $: modelReady = status.prepared && JSON.stringify(config) === JSON.stringify(status.config);
  $: prepared = modelReady && status.prepared_prompt === prompt.trim();
  $: canStart = connected && prepared && status.state === 'ready' && cameraReady && !!referenceSource && !!prompt.trim() && !busy;
  $: referenceSource = ['starting', 'running', 'stopping'].includes(status.state) && capturedReference ? capturedReference : cameraReady ? camera : null;

  function send(data: object) {
    if (socket?.readyState !== WebSocket.OPEN) throw new Error('Disconnected from the server');
    socket.send(JSON.stringify(data));
  }
  function clearPlayback() {
    for (const frame of playback) frame.bitmap.close();
    playback = []; playbackClock = new PlaybackClock(); shown = 0;
    if (output) output.getContext('2d')!.clearRect(0, 0, output.width, output.height);
  }
  function changeSession(sid: string | null) {
    if (sid === session) return;
    session = sid; credits = 0; sequence = 0; nextCapture = 0;
    uploadWidth = uploadHeight = 0;
    if (!sid) capturedReference = null;
    captureCount = playbackCount = 0; captureFps = playbackFps = latency = 0;
    clearPlayback();
  }
  function connect() {
    if (socket && socket.readyState < 2) return;
    error = '';
    let initialStatus = true;
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/ws`);
    socket.binaryType = 'arraybuffer';
    socket.onopen = () => connected = true;
    socket.onclose = () => { connected = false; busy = false; status = { ...status, state: 'disconnected', message: 'Disconnected from the server' }; changeSession(null); };
    socket.onerror = () => error = 'Unable to connect to the server. Check its logs.';
    socket.onmessage = event => {
      if (typeof event.data === 'string') {
        const data = JSON.parse(event.data);
        if (data.type === 'status') {
          if (initialStatus) {
            captureTargetFps = data.driving_fps;
            if (data.config) ({ resolution, fp8, decoder, dual_gpu: dualGpu } = data.config);
            if (data.prepared_prompt) prompt = data.prepared_prompt;
          }
          initialStatus = false;
          status = data; changeSession(data.session_id);
          if (!['preparing', 'starting', 'stopping'].includes(data.state)) busy = false;
        } else if (data.type === 'credit' && data.session_id === session) {
          credits += data.count;
        } else if (data.type === 'error') { error = data.message; busy = false; }
      } else {
        const { header, payload } = unpack(event.data);
        if (header.session_id !== session) return;
        decodeChain = decodeChain.then(async () => {
          if (header.session_id !== session) return;
          const bitmap = await createImageBitmap(payload);
          if (header.session_id !== session) { bitmap.close(); return; }
          if (header.chunk_frame === 0) playbackClock.observe(performance.now(), header.chunk_frames, playback.length);
          playback.push({ bitmap, header });
        }).catch(exc => { error = `Could not decode an output frame: ${exc.message}`; });
      }
    };
  }
  function setCaptureFps() {
    try { send({ type: 'fps', fps: captureTargetFps }); } catch (exc) { error = String(exc); }
  }
  async function openCamera() {
    try {
      error = '';
      if (!navigator.mediaDevices) throw new Error('Camera access requires localhost or HTTPS.');
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
      camera.srcObject = stream;
      await camera.play(); cameraReady = true;
      stream.getVideoTracks()[0].onended = () => { cameraReady = false; if (running) stop(); };
    } catch (exc) { error = String(exc); }
  }
  async function prepareModel() {
    error = ''; busy = true; preparing = 'model';
    try { send({ type: 'prepare_model', config }); } catch (exc) { error = String(exc); busy = false; }
  }
  async function preparePrompt() {
    error = ''; busy = true; preparing = 'prompt';
    try { send({ type: 'prepare_prompt', config, prompt: prompt.trim() }); } catch (exc) { error = String(exc); busy = false; }
  }
  async function cameraReference() {
    const blob = await jpeg(camera, undefined, true);
    const url = URL.createObjectURL(blob);
    try {
      const image = new Image(); image.src = url; await image.decode();
      capturedReference = image;
    } finally { URL.revokeObjectURL(url); }
    return blob;
  }
  async function start() {
    busy = true; error = '';
    try {
      const payload = await cameraReference();
      socket!.send(packet({ type: 'start', prompt: prompt.trim(), reference_mode: 'camera', crop: frameCrop, seed: -1 }, payload));
    } catch (exc) { error = String(exc); busy = false; }
  }
  function stop() {
    busy = true; error = ''; credits = 0; clearPlayback();
    try { send({ type: 'stop' }); } catch (exc) { error = String(exc); busy = false; }
  }
  async function capture(now: number) {
    if (!running || busy || !session || !cameraReady || credits <= 0 || encoding || now < nextCapture || !socket || socket.bufferedAmount > 2 * 1024 * 1024) return;
    const sid = session;
    const interval = 1000 / Math.max(1, captureTargetFps);
    nextCapture = nextCapture ? nextCapture + interval : now + interval;
    if (nextCapture <= now) nextCapture = now + interval;
    credits--; encoding = true;
    const id = sequence++;
    try {
      const blob = await jpeg(camera, frameCrop, true, captureCanvas);
      if (sid === session && running && !busy && socket.readyState === WebSocket.OPEN) {
        socket.send(packet({ type: 'frame', session_id: sid, frame_id: id, capture_ms: now }, blob));
        captureCount++;
        uploadWidth = captureCanvas.width; uploadHeight = captureCanvas.height;
      }
    } catch (exc) { error = String(exc); if (sid === session) credits++; }
    finally { encoding = false; }
  }
  function videoFrame(now: number) {
    if (stopped) return;
    void capture(now);
    camera.requestVideoFrameCallback(videoFrame);
  }
  function tick(now: number) {
    if (stopped) return;
    if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype) && camera.currentTime !== lastVideoTime) {
      lastVideoTime = camera.currentTime; void capture(now);
    }
    const frame = playback[0];
    if (frame && !busy && playbackClock.ready(now)) {
      playback.shift();
      if (output.width !== frame.bitmap.width || output.height !== frame.bitmap.height) {
        output.width = frame.bitmap.width; output.height = frame.bitmap.height;
      }
      output.getContext('2d')!.drawImage(frame.bitmap, 0, 0); frame.bitmap.close();
      latency = (now - frame.header.capture_ms) / 1000; playbackCount++; shown++;
      send({ type: 'played', session_id: session, output_id: frame.header.output_id });
    }
    if (now - statsTime >= 2000) {
      const seconds = (now - statsTime) / 1000;
      captureFps = captureCount / seconds; playbackFps = playbackCount / seconds;
      captureCount = playbackCount = 0; statsTime = now;
    }
    raf = requestAnimationFrame(tick);
  }
  onMount(() => {
    captureCanvas = document.createElement('canvas');
    connect(); statsTime = performance.now(); raf = requestAnimationFrame(tick);
    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) camera.requestVideoFrameCallback(videoFrame);
    return () => { stopped = true; cancelAnimationFrame(raf); socket?.close(); stream?.getTracks().forEach(track => track.stop()); clearPlayback(); };
  });

  const letters = [
    ['E', '#b9a7e5'], ['d', '#f2acc5'], ['i', '#f4d582'], ['t', '#83c7b9'], ['a', '#efaa98'],
    ['L', '#8dbed8'], ['i', '#b9a7e5'], ['v', '#f4d582'], ['e', '#83c7b9'], ['!', '#f2acc5'],
  ];
</script>
<svelte:head><title>EditaLive! · Live studio</title></svelte:head>
<video bind:this={camera} muted playsinline class="camera-source"></video>
<div class="page">
  <header class="site-header">
    <div class="brand">
      <a class="lab-brand" href="https://gvclab.github.io/" target="_blank" rel="noreferrer" aria-label="Visit the GVC Lab website"><img src="/branding/gvc-live-research.png" alt="GVC LIVE Research" width="1408" height="434" /></a>
      <span class="brand-divider" aria-hidden="true"></span>
      <h1 aria-label="EditaLive!">{#each letters as [letter, color]}<span aria-hidden="true" style={`--letter:${color}`}>{letter}</span>{/each}</h1>
    </div>
    <div class="header-links">
      <a class="project-link" href="https://huai-chang.github.io/EditaLive/" target="_blank" rel="noreferrer">Project page <Icon name="external" size={14} /></a>
      <a href="https://github.com/GVCLab/EditaLive" target="_blank" rel="noreferrer">GitHub <Icon name="external" size={14} /></a>
      <span class="connection" class:online={connected}><i></i>{connected ? 'Connected' : 'Disconnected'}</span>
      {#if !connected}<button class="small" on:click={connect}>Reconnect</button>{/if}
    </div>
  </header>
  <main>
    <div class="views">
      <section class="panel camera-section" aria-labelledby="camera-title">
        <div class="view-heading">
          <h2 id="camera-title" class="sticker pink"><Icon name="camera" /> Camera view</h2>
          <button class="small camera-toggle" class:ready={cameraReady} on:click={openCamera} disabled={cameraReady || locked}>{cameraReady ? 'Camera is on' : 'Enable camera'}</button>
        </div>
        <Cropper source={cameraReady ? camera : null} {aspect} previewAspect={16 / 9} bind:crop={frameCrop} disabled={locked} mirrored label="Camera">
          <div class="placeholder">
            <div class="illustration camera-illustration" aria-hidden="true"><Icon name="camera" size={38} /><span class="little-star">✦</span></div>
            <strong>Your camera goes here</strong>
            <p>Enable your camera, then find your frame.</p>
            <span class="mini-tag">Mirrored · Video only</span>
          </div>
        </Cropper>
      </section>
      <section class="panel output-section" aria-labelledby="output-title">
        <div class="view-heading">
          <h2 id="output-title" class="sticker lilac"><Icon name="sparkles" /> Live output</h2>
          <span class="live-label" class:active={running} role="status" aria-label={status.message} title={status.message}><i></i>{running ? 'LIVE' : status.state === 'preparing' ? 'PREPARING' : status.state === 'starting' ? 'STARTING' : 'STANDBY'}</span>
        </div>
        <div class="output-wrap" class:has-frames={shown > 0}>
          <canvas bind:this={output} aria-label="Generated video" data-frames={shown} />
          {#if !shown}<div class="placeholder">
            <div class="illustration output-illustration" aria-hidden="true"><Icon name="sparkles" size={39} /><span class="little-star">♥</span></div>
            <strong>{running ? 'Making a little magic…' : 'Your next look starts here'}</strong>
            <p>{modelPreparing ? 'First-time preparation may take a few minutes.' : running ? 'The first frames are on their way.' : 'Write a prompt. Make it yours. Go live.'}</p>
            <span class="mini-tag">{running ? 'Generating first frames' : 'Ready when you are'}</span>
          </div>{/if}
        </div>
      </section>
    </div>
    <div class="controls-grid">
      <section class="panel settings-section" aria-labelledby="settings-title">
        <div class="panel-heading"><h2 id="settings-title" class="sticker mint"><Icon name="sliders" /> Run settings</h2><span class="panel-index">01</span></div>
        <div class="form-grid">
          <label>Resolution<select bind:value={resolution} disabled={locked}><option value={384}>672 × 384</option><option value={480}>832 × 480</option></select></label>
          <label>Precision<select bind:value={fp8} disabled={locked}><option value={true}>FP8</option><option value={false}>BF16</option></select></label>
          <label>Decoder<select bind:value={decoder} disabled={locked}><option value="flash">Flash-VAED</option><option value="original">Wan VAE</option></select></label>
          <label title="Runs pose extraction and the VAE on a second card, so the diffusion card is free to start the next chunk.">GPUs<select aria-label="GPUs" bind:value={dualGpu} disabled={locked}><option value={false}>Single card</option><option value={true}>Two cards</option></select></label>
        </div>
        {#if modelPreparing}
          <ModelPreparation progress={status.preparation} />
        {:else}
          <button class="prepare model-prepare" class:prepared={modelReady} on:click={prepareModel} disabled={!connected || locked || modelReady}><Icon name={modelReady ? 'check' : 'sliders'} size={16} />{modelReady ? 'Model ready' : 'Prepare model'}</button>
        {/if}
      </section>
      <section class="panel edit-section" aria-labelledby="edit-title">
        <div class="panel-heading"><h2 id="edit-title" class="sticker butter"><Icon name="edit" /> Reference & edit</h2><span class="panel-index">02</span></div>
        <div class="edit-grid">
          <div class="reference">
            <span class="field-label">Camera reference</span>
            <ReferencePreview source={referenceSource} {aspect} crop={frameCrop} mirrored={referenceSource === camera} />
          </div>
          <div class="prompt-editor">
            <label class="prompt">Prompt<textarea bind:value={prompt} disabled={locked || !modelReady} rows="3" maxlength="4000" placeholder="Describe the appearance you want…" /></label>
            <div class="edit-actions">
              <button class="prepare" class:prepared on:click={preparePrompt} disabled={!connected || locked || !modelReady || !prompt.trim() || prepared}><Icon name={prepared ? 'check' : 'sparkles'} size={16} />{preparing === 'prompt' ? 'Preparing prompt…' : prepared ? 'Prompt ready' : 'Prepare prompt'}</button>
              <button class="primary" on:click={start} disabled={!canStart}><Icon name="play" size={16} />Start editing</button>
              <button on:click={stop} disabled={!connected || busy || !['running', 'error'].includes(status.state) || !status.prepared} title="Reset session to edit the prompt or camera crop"><Icon name="reset" size={17} />Reset</button>
            </div>
          </div>
        </div>
      </section>
      <section class="panel stats-section" aria-labelledby="stats-title">
        <div class="panel-heading"><h2 id="stats-title" class="sticker blush"><Icon name="activity" /> Live stats</h2></div>
        <dl class="metrics">
          <div class="generation-metric"><dt>Generation</dt><dd>{(status.generation_fps || 0).toFixed(1)}<small>fps</small></dd></div>
          <div><dt>Capture</dt><dd>{captureFps.toFixed(1)}<small>fps</small></dd></div>
          <div><dt>Playback</dt><dd>{playbackFps.toFixed(1)}<small>fps</small></dd></div>
          <div><dt>Display latency</dt><dd>{latency.toFixed(2)}<small>s</small></dd></div>
        </dl>
        <label class="capture-rate" title="Target webcam capture rate. Adjustable while editing."><span>Capture FPS <b>{captureTargetFps}<small>fps</small></b></span><input aria-label="Capture FPS" type="range" min="8" max="25" step="1" bind:value={captureTargetFps} on:change={setCaptureFps} disabled={!connected} /></label>
        <details class="stream-details">
          <summary>Stream details</summary>
          <dl class="buffer-metrics">
            <div><dt>Upload size</dt><dd>{uploadWidth ? `${uploadWidth} × ${uploadHeight}` : '—'}</dd></div>
            <div><dt title="Input frames waiting for processing.">Input queue</dt><dd>{status.input_frames || 0} / 24</dd></div>
            <div><dt title="Frames being generated or awaiting playback confirmation.">Output slots</dt><dd>{status.output_frames || 0} / 24</dd></div>
            <div><dt title="Input frames consumed per model input frame, up to 1.5×.">Catch-up</dt><dd>{(status.catchup_ratio || 1).toFixed(2)}×</dd></div>
          </dl>
        </details>
      </section>
    </div>
  </main>
  {#if error}<div class="error" role="alert"><span>{error}</span><button on:click={() => error = ''} aria-label="Dismiss error">×</button></div>{/if}
</div>
<style>
  @font-face { font-family:Fredoka; src:url('/fonts/Fredoka-Variable.ttf') format('truetype'); font-weight:300 700; font-style:normal; font-display:swap; }
  :global(*) { box-sizing:border-box; }
  :global(body) { margin:0; background:#f6f2eb; background-image:linear-gradient(#75679e09 1px,transparent 1px),linear-gradient(90deg,#75679e09 1px,transparent 1px); background-size:36px 36px; color:#39313f; font-family:"Avenir Next","Segoe UI",Helvetica,Arial,sans-serif; font-size:14px; }
  :global(button), :global(input), :global(select), :global(textarea) { font:inherit; }
  :global(button:focus-visible), :global(a:focus-visible), :global(input:focus-visible), :global(select:focus-visible), :global(textarea:focus-visible), :global(summary:focus-visible) { outline:3px solid #9580c3; outline-offset:3px; }
  .camera-source { position:fixed; width:1px; height:1px; opacity:0; pointer-events:none; }
  .page { max-width:1800px; margin:auto; padding:0 24px 14px; }
  .site-header { min-height:68px; display:flex; align-items:center; justify-content:space-between; gap:20px; padding:12px 2px 16px; }
  .brand { display:flex; align-items:center; gap:16px; }
  h1 { display:flex; margin:0; font-family:Fredoka,sans-serif; font-size:38px; font-weight:650; letter-spacing:-1.4px; line-height:1; filter:drop-shadow(0 3px 1px #60484226); }
  h1 span { color:var(--letter); -webkit-text-stroke:.6px #79658845; paint-order:stroke fill; text-shadow:-2px -2px 0 #fffdf8,0 -3px 0 #fffdf8,2px -2px 0 #fffdf8,-3px 0 0 #fffdf8,3px 0 0 #fffdf8,-2px 2px 0 #fffdf8,0 3px 0 #fffdf8,2px 2px 0 #fffdf8; }
  h1 span:nth-child(2n) { transform:rotate(3deg) translateY(1px); } h1 span:nth-child(3n) { transform:rotate(-4deg); }
  .lab-brand { flex-shrink:0; border-radius:4px; }
  .lab-brand img { display:block; width:auto; height:26px; }
  .brand-divider { width:1px; height:28px; flex-shrink:0; background:#d8cfdb; }
  .header-links { display:flex; align-items:center; gap:22px; font-size:12px; }
  a { color:#776982; text-decoration:none; display:inline-flex; align-items:center; gap:6px; } a:hover { color:#47385f; text-decoration:underline; }
  .connection { display:flex; align-items:center; gap:7px; color:#8b6465; } .connection.online { color:#4e7c72; }
  i { display:inline-block; width:7px; height:7px; flex-shrink:0; border-radius:50%; background:#c1b2c3; }
  .connection i { background:#cd8a8f; } .connection.online i { background:#6ca797; box-shadow:0 0 0 4px #6ca79715; }
  main { display:grid; gap:16px; }
  .views { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
  .controls-grid { display:grid; grid-template-columns:minmax(290px,.95fr) minmax(440px,1.4fr) minmax(240px,.9fr); gap:16px; align-items:stretch; }
  .panel { min-width:0; background:#fffdf8; border:3px solid #fffefa; border-radius:19px; padding:14px; box-shadow:0 0 0 1px #d9d0dc,0 4px 0 #d9d0dc88,0 9px 22px #65547908; }
  .views .panel { position:relative; padding:5px; overflow:hidden; }
  .view-heading { position:absolute; z-index:3; top:15px; left:15px; right:15px; display:flex; align-items:center; justify-content:space-between; gap:12px; pointer-events:none; }
  .view-heading .sticker { font-size:12px; gap:5px; padding:4px 7px; border-width:2px; border-radius:9px; transform:none; background:#f8edf4ed; }
  .view-heading .lilac { background:#eee7f7ed; }
  .view-heading :global(svg) { width:14px; height:14px; }
  .view-heading button.small { min-height:27px; padding:4px 8px; font-size:11px; background:#fffdf8e8; }
  .view-heading button:enabled { pointer-events:auto; }
  .panel-heading { min-height:32px; display:flex; align-items:center; justify-content:space-between; gap:12px; margin:0 0 12px; }
  .panel-heading .sticker { transform:none; }
  h2 { margin:0; font-family:Fredoka,sans-serif; font-size:16px; font-weight:530; white-space:nowrap; }
  .sticker { display:inline-flex; align-items:center; gap:8px; padding:5px 10px; border:3px solid #fffefa; border-radius:12px; box-shadow:inset 0 0 0 1px #75679e24,0 2px 0 #66527522,0 4px 9px #6652750a; transform:rotate(-2deg); }
  .pink { background:#f8d8df; color:#895267; } .lilac { background:#e7ddf4; color:#6c588d; transform:rotate(1.5deg); }
  .mint { background:#d9eee8; color:#447c71; } .butter { background:#ffebae; color:#8b692a; transform:rotate(1deg); } .blush { background:#f4e1eb; color:#8a5d79; }
  .panel-index { font-family:Fredoka,sans-serif; color:#b1a4b7; font-size:17px; padding-right:3px; }
  button { display:inline-flex; align-items:center; justify-content:center; gap:7px; min-height:34px; border:1px solid #d9cedf; border-radius:10px; padding:8px 11px; background:#fffdfb; color:#655772; cursor:pointer; font-family:Fredoka,sans-serif; font-weight:450; transition:background .15s,box-shadow .15s; }
  button:hover:enabled { background:#f1eaf8; border-color:#b9a5cd; box-shadow:0 2px 0 #d9cedf; }
  button:disabled { cursor:default; color:#a69bab; background:#f3efef; border-color:#e3dce3; }
  button.small { min-height:32px; padding:6px 11px; font-size:12px; }
  .camera-toggle.ready { color:#6b8d80; background:#eaf4ee; border-color:#d3e2d9; }
  .live-label { display:flex; align-items:center; gap:7px; color:#a297ad; font-size:10px; font-weight:700; letter-spacing:1.2px; padding:6px 8px; border:1px solid #e2d8e6; border-radius:8px; background:#fffdf8e8; }
  .live-label.active { color:#ba5975; } .live-label.active i { background:#e57896; box-shadow:0 0 0 4px #e578961c; }
  .output-wrap { position:relative; width:100%; aspect-ratio:16 / 9; display:grid; place-items:center; overflow:hidden; border:1px solid #ded5e4; border-radius:13px; background-color:#f1ecf4; background-image:linear-gradient(#8f78a308 1px,transparent 1px),linear-gradient(90deg,#8f78a308 1px,transparent 1px); background-size:28px 28px; }
  .output-wrap.has-frames { background:#171723; }
  .output-wrap canvas { position:absolute; width:100%; height:100%; object-fit:contain; }
  .placeholder { z-index:1; text-align:center; padding:20px; }
  .placeholder strong { display:block; font-family:Fredoka,sans-serif; font-size:clamp(18px,1.6vw,26px); font-weight:500; color:#726081; }
  .placeholder p { font-size:12px; color:#81718b; margin:9px 0 17px; line-height:1.5; }
  .illustration { display:grid; place-items:center; position:relative; width:80px; height:76px; margin:0 auto 24px; border:5px solid #fffdf8; border-radius:24px; box-shadow:inset 0 0 0 1px #75679e24,0 4px 0 #74608622,0 9px 18px #74608612; }
  .camera-illustration { background:#f5d2de; color:#ad7088; transform:rotate(-9deg); }
  .output-illustration { background:#ded1f0; color:#9877bf; transform:rotate(9deg); }
  .little-star { position:absolute; right:-19px; top:-17px; color:#e0b75f; font-size:31px; -webkit-text-stroke:3px #fffdf8; paint-order:stroke fill; }
  .output-illustration .little-star { color:#e7a3b8; font-size:29px; }
  .mini-tag { display:inline-flex; border:1px solid #ded2e4; padding:5px 10px; border-radius:20px; font-size:10px; color:#806d8b; background:#fffdf85c; letter-spacing:.2px; }
  label,.field-label { display:block; color:#796680; font-size:11px; font-weight:500; }
  .field-label { margin-bottom:7px; }
  .form-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
  select,textarea { display:block; width:100%; min-width:0; margin-top:7px; border:1px solid #ded5e2; border-radius:9px; background:#fffefb; color:#5d4f69; padding:8px; font-size:12px; }
  select { height:35px; }
  textarea { flex:1; height:64px; min-height:64px; line-height:1.6; resize:vertical; }
  textarea::placeholder { color:#b0a2b8; }
  select:disabled,textarea:disabled { color:#a69bab; background:#f3efef; }
  .capture-rate { margin-top:8px; }
  .capture-rate>span { display:flex; justify-content:space-between; align-items:center; }
  .capture-rate b { color:#567f73; font-family:Fredoka,sans-serif; font-size:17px; font-weight:500; }
  small { font-size:10px; font-weight:400; margin-left:4px; color:#817189; }
  input[type=range] { width:100%; margin:5px 0 0; accent-color:#82b2a3; cursor:pointer; }
  input[type=range]:disabled { cursor:default; }
  .edit-section { container-type:inline-size; }
  .edit-grid { display:grid; grid-template-columns:minmax(120px,1fr) minmax(0,2fr); gap:12px; }
  .reference,.prompt-editor { min-width:0; }
  .reference,.prompt-editor,.prompt { display:flex; flex-direction:column; }
  .prompt { flex:1; }
  .edit-actions { display:grid; grid-template-columns:minmax(0,1fr) auto auto; gap:6px; margin-top:10px; }
  .edit-actions button { font-size:11px; padding:6px 8px; gap:5px; white-space:nowrap; }
  .model-prepare { width:100%; margin-top:14px; }
  .prepare { flex:1; background:#eee5f7; border-color:#d9cbe6; color:#7b5d9a; }
  .prepare.prepared { background:#edf5ec; border-color:#d4e3d3; color:#70956c; }
  .primary { background:#d3eae1; color:#386f60; border-color:#b3d2c3; box-shadow:0 2px 0 #b7d1c355; }
  .primary:hover:enabled { background:#bfdfd1; border-color:#8bbba4; box-shadow:0 2px 0 #a2c8b5; }
  .metrics { margin:0; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px 12px; }
  .metrics>div { min-width:0; display:flex; align-items:baseline; justify-content:space-between; gap:8px; padding:0 0 6px; border-bottom:1px solid #e9e1eb; }
  .metrics dt { color:#7f6b87; font-size:11px; } .metrics dd { margin:0; font-family:Fredoka,sans-serif; font-size:22px; color:#796188; font-weight:400; line-height:1.3; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .generation-metric dt { color:#9572a5; } .generation-metric dd { font-size:24px; line-height:1.3; color:#9473aa; }
  .generation-metric small { font-family:"Segoe UI",sans-serif; font-size:11px; margin-left:7px; }
  .stream-details { margin-top:8px; color:#85758e; font-size:10px; }
  summary { cursor:pointer; padding:3px 0; width:fit-content; }
  summary:hover { color:#655772; }
  .buffer-metrics { margin:8px 0 0; display:grid; gap:7px; font-size:11px; }
  .buffer-metrics>div { display:flex; justify-content:space-between; gap:12px; }
  .buffer-metrics dd { margin:0; font-variant-numeric:tabular-nums; }
  .error { position:fixed; bottom:24px; left:50%; transform:translateX(-50%); display:flex; align-items:center; gap:20px; max-width:90%; background:#fff0f0; border:3px solid #fffdfa; border-radius:14px; padding:12px 16px; box-shadow:0 0 0 1px #e8bfcc,0 5px 20px #7d3d521f; color:#a25571; font-size:13px; z-index:10; overflow-wrap:anywhere; }
  .error button { padding:0 5px; border:none; background:transparent; color:inherit; font-size:22px; }
  @media(min-width:1100px) {
    .page { min-height:100dvh; }
    .placeholder { padding:10px; }
    .placeholder strong { font-size:21px; }
    .placeholder p { margin:7px 0 10px; }
    .illustration { width:60px; height:58px; margin-bottom:14px; border-width:4px; border-radius:19px; }
    textarea { resize:none; }
  }
  @media(min-width:1100px) and (max-height:800px) {
    .site-header { min-height:56px; padding:8px 2px 12px; } h1 { font-size:33px; }
    main { gap:12px; } .panel { padding:11px; } .panel-heading { margin-bottom:8px; }
    .sticker { padding:4px 9px; }
    .model-prepare { margin-top:10px; }
    .illustration { width:46px; height:44px; margin-bottom:10px; }
    .illustration :global(svg) { width:29px; height:29px; }
    .placeholder strong { font-size:19px; } .placeholder p { font-size:11px; }
    .mini-tag { display:none; }
  }
  @media(max-width:1099px) {
    .page { padding:0 18px 14px; }
    .controls-grid { grid-template-columns:minmax(0,1fr) minmax(0,1.6fr); }
    .stats-section { grid-column:1 / -1; display:grid; grid-template-columns:145px 1fr; column-gap:18px; }
    .stats-section .panel-heading { grid-row:1 / 4; align-self:center; margin:0; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:15px; }
    .metrics>div,.metrics .generation-metric { display:block; border:0; padding:0; }
    .metrics dd,.generation-metric dd { font-size:23px; line-height:1.5; }
    .capture-rate { grid-column:2; }
    .stream-details { grid-column:2; }
  }
  @media(max-width:760px) {
    .page { padding:0 14px 14px; } .site-header { min-height:70px; padding:15px 2px; flex-wrap:wrap; gap:10px; } h1 { font-size:33px; }
    .brand { gap:12px; } .lab-brand img { height:22px; } .brand-divider { height:24px; }
    .project-link { display:none; } .header-links { gap:8px; font-size:10px; }
    .views,.controls-grid { grid-template-columns:1fr; }
    .view-heading { top:12px; left:12px; right:12px; gap:6px; }
    .view-heading .sticker { font-size:11px; }
    .panel { padding:13px; border-radius:18px; } h2 { font-size:15px; }
    .placeholder strong { font-size:19px; } .illustration { width:58px; height:55px; margin-bottom:13px; }
    .placeholder { padding:12px; } .placeholder p { font-size:11px; margin:7px 0 10px; } .mini-tag { font-size:9px; }
    .stats-section { grid-column:auto; display:block; } .stats-section .panel-heading { margin-bottom:16px; }
    .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
    .metrics>div { border-bottom:1px solid #e9e1eb; padding-bottom:8px; }
    .connection { gap:6px; } .panel-index { font-size:15px; }
  }
  @container(max-width:470px) {
    .edit-grid { grid-template-columns:100px minmax(0,1fr); }
    .edit-actions { grid-template-columns:minmax(0,1fr) auto; }
    .edit-actions .prepare { grid-column:1 / -1; }
  }
  @media(prefers-reduced-motion:reduce) { button { transition:none; } }
</style>
