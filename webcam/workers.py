import os
import queue
import threading
import time
import traceback
from pathlib import Path


class JpegEncoder:
    """Encode completed chunks without blocking the generator."""

    def __init__(self, events):
        self.events = events
        self.queue = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, event, result, ready):
        self.queue.put((event, result, ready))

    def drop(self, session_id):
        """Discard one session, or all queued chunks when session_id is None."""
        pending = []
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if session_id is not None and item[0].get("session_id") != session_id:
                pending.append(item)
        for item in pending:
            self.queue.put(item)

    def stop(self):
        self.drop(None)
        self.queue.put(None)
        self.thread.join(timeout=5)

    def _run(self):
        from .images import encode_jpeg
        while True:
            item = self.queue.get()
            if item is None:
                return
            event, result, ready = item
            try:
                ready.synchronize()
                event["frames"] = [encode_jpeg(frame) for frame in result.numpy()]
                event["completed_at"] = time.perf_counter()
                self.events.put(event)
            except Exception as exc:
                self.events.put(dict(type="error", worker=event.get("worker"), job=event.get("job"),
                                     session_id=event.get("session_id"),
                                     error=str(exc) or type(exc).__name__,
                                     traceback=traceback.format_exc()))


class ConditionEncoder:
    """Prepare one future chunk on the auxiliary GPU."""

    def __init__(self, pipeline, commands, events, shared, size):
        self.pipeline, self.commands, self.events = pipeline, commands, events
        self.shared, self.size = shared, size
        self.session = None
        self.session_id = None
        self.results = {}
        self.lock = threading.Lock()
        self.barrier = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.barrier.wait(1800):
            raise TimeoutError("Condition warmup timed out.")
        if self.error is not None:
            raise self.error

    def start(self, session, session_id):
        self.session, self.session_id = session, session_id
        self.next_chunk = 0

    def take(self, job):
        import torch
        with self.lock:
            latents, ready = self.results.pop(job)
        stream = torch.cuda.current_stream(self.pipeline.device)
        stream.wait_event(ready)
        latents.record_stream(stream)
        return latents

    def close(self):
        self.session_id = None
        self.barrier.clear()
        self.commands.put(dict(type="barrier"))
        if not self.barrier.wait(180):
            raise TimeoutError("Condition shutdown timed out.")
        self.session = None

    def stop(self):
        self.close()
        self.commands.put(dict(type="shutdown"))
        self.thread.join(timeout=5)

    def _run(self):
        import torch
        p = self.pipeline
        try:
            torch.cuda.set_device(p.device)
            stream = torch.cuda.Stream(device=p.vae_device)
            transfer = torch.cuda.Stream(device=p.device)
            width, height = self.size
            with torch.cuda.stream(stream):
                p.vae.warmup_stream([9, 12], p.chunk_size, (height, width),
                                   (height // 8, width // 8), encode_dtype=torch.bfloat16)
            stream.synchronize()
        except Exception as exc:
            self.error = exc
            self.barrier.set()
            return
        self.barrier.set()
        while True:
            command = self.commands.get()
            action = command["type"]
            if action == "shutdown":
                return
            if action == "barrier":
                stream.synchronize()
                transfer.synchronize()
                with self.lock:
                    self.results.clear()
                self.barrier.set()
                continue
            sid, job = command["session_id"], command["job"]
            if sid != self.session_id:
                continue
            try:
                batch = command["batch"]
                if batch["chunk_id"] != self.next_chunk:
                    raise ValueError("Pose chunks must be encoded in order.")
                pose, _ = self.shared.read(command["slot"], len(batch["capture_ms"]))
                started = time.perf_counter()
                with torch.no_grad(), torch.cuda.stream(stream):
                    latents = self.session.encode_pose(pose, len(pose))
                    encoded = torch.cuda.Event()
                    encoded.record(stream)
                    with torch.cuda.stream(transfer):
                        transfer.wait_event(encoded)
                        latents = latents.to(p.device, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record(transfer)
                with self.lock:
                    self.results[job] = (latents, ready)
                self.next_chunk += 1
                self.events.put(dict(type="conditioned", worker="generation", job=job,
                                     session_id=sid, slot=command["slot"], batch=batch,
                                     condition_submit_seconds=time.perf_counter() - started))
            except Exception as exc:
                self.events.put(dict(type="error", worker="generation", job=job, session_id=sid,
                                     error=str(exc) or type(exc).__name__, traceback=traceback.format_exc()))


def worker(kind, config_data, paths_data, commands, events, shared=None, condition_commands=None):
    os.environ.setdefault("WAN_ORT_INTRA_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from .config import RuntimeConfig, ModelPaths
    config, paths = RuntimeConfig(**config_data), ModelPaths(**paths_data)
    cache = (Path(paths.compile_cache) /
             f"{config.resolution}-{config.orientation}-{config.fp8}-{config.decoder}-{config.dual_gpu}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    session = None
    prepared = None
    encoder = None
    conditioner = None

    def progress(message):
        events.put(dict(type="progress", worker=kind, message=message))

    try:
        import torch
        torch.set_num_threads(4)
        main, aux = paths.resolve_devices(config)
        index = main if kind == "generation" else aux
        for wanted in {index, aux}:
            if not 0 <= wanted < torch.cuda.device_count():
                raise ValueError(f"CUDA device {wanted} is unavailable; "
                                 f"visible devices: {torch.cuda.device_count()}.")
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
        if kind == "generation":
            from editalive.compile_config import configure_compile
            configure_compile(paths.compile)
            from editalive.configs import editalive_14B
            from editalive.pipeline.editalive_streaming import EditaLiveStreamingPipeline
            progress("Loading model weights")
            pipeline = EditaLiveStreamingPipeline.from_pretrained(
                paths.checkpoint, editalive_14B, device=device,
                vae_device=torch.device(f"cuda:{aux}"),
                lora_paths=paths.loras, fp8=config.fp8,
                fast_decode=config.decoder == "flash", fast_decoder_pth=paths.fast_decoder,
                vsa_sparsity=0.75, vsa_backend="kernel")
            width, height = config.size
            pipeline.prepare_stream_model(height, width, fast_decode=config.decoder == "flash",
                                          progress_callback=progress)
        else:
            from .extractor import OnlineExtractor
            from .flow import CHUNK, FIRST_CHUNK
            progress("Loading tracking models")
            extractor = OnlineExtractor(paths.checkpoint, device, config.size)
            progress("Warming up motion tracking")
            extractor.warmup((FIRST_CHUNK, CHUNK))
        def graph_count():
            return torch._dynamo.utils.counters["stats"]["unique_graphs"] if kind == "generation" else 0
        if kind == "generation":
            encoder = JpegEncoder(events)
            if config.dual_gpu and condition_commands is not None:
                progress("Warming up dual-GPU pipeline")
                conditioner = ConditionEncoder(pipeline, condition_commands, events, shared, config.size)
        events.put(dict(type="ready", worker=kind, compiled_graphs=graph_count()))
    except Exception as exc:
        events.put(dict(type="fatal", worker=kind, error=str(exc) or type(exc).__name__, traceback=traceback.format_exc()))
        return
    while True:
        command = commands.get()
        action, job = command["type"], command["job"]
        sid = command.get("session_id")
        queue_seconds = time.perf_counter() - command["queued_at"]
        try:
            if action == "shutdown":
                if conditioner is not None:
                    conditioner.stop()
                if encoder is not None:
                    encoder.stop()
                return
            if action == "close":
                if conditioner is not None:
                    conditioner.close()
                if encoder is not None:
                    encoder.drop(sid)
                if session is not None:
                    session.close()
                    session = None
                if kind == "extraction":
                    extractor.reset()
                events.put(dict(type="closed", worker=kind, job=job))
            elif action == "prepare":
                width, height = config.size
                events.put(dict(type="progress", worker=kind, message="Encoding prompt"))
                prepared = pipeline.prepare_stream(command["prompt"], height, width,
                                                   fast_decode=config.decoder == "flash")
                torch.cuda.synchronize(device)
                events.put(dict(type="prepared", worker=kind, job=job))
            elif action == "start":
                if prepared is None or prepared["prompt"] != command["prompt"]:
                    raise RuntimeError("Prepare this prompt before starting.")
                if encoder is not None:
                    encoder.drop(session_id if session is not None else None)
                if session is not None:
                    if conditioner is not None:
                        conditioner.close()
                    session.close()
                seed = command["seed"]
                generator = torch.Generator(device=device).manual_seed(seed)
                session = pipeline.stream_start(command["reference"], command["prompt"],
                                                generator=generator, fast_decode=config.decoder == "flash",
                                                prepared=prepared)
                session_id = sid
                if conditioner is not None:
                    conditioner.start(session, sid)
                generation_started = last_generation_end = None
                generated_frames = 0
                busy_seconds = 0.0
                events.put(dict(type="started", worker=kind, job=job, session_id=sid))
            elif action == "extract":
                start = time.perf_counter()
                pose, face = extractor(command["batch"]["frames"])
                batch = {k: v for k, v in command["batch"].items() if k != "frames"}
                batch["frame_ids"] = [f[0] for f in command["batch"]["frames"]]
                batch["capture_ms"] = [f[1] for f in command["batch"]["frames"]]
                batch["extract_seconds"] = time.perf_counter() - start
                batch["queue_seconds"] = queue_seconds
                if shared is not None:
                    shared.write(command["slot"], pose, face)
                    payload = dict(slot=command["slot"])
                else:
                    payload = dict(pose=pose, face=face)
                events.put(dict(type="extracted", worker=kind, job=job, session_id=sid,
                                batch=batch, **payload, extract_seconds=time.perf_counter() - start))
            elif action == "generate":
                if session is None or sid != session_id:
                    raise RuntimeError("Stale generation session.")
                start = time.perf_counter()
                if generation_started is None:
                    generation_started = start
                idle_seconds = 0.0 if last_generation_end is None else start - last_generation_end
                noise = torch.randn(session.noise_shape, device=device, dtype=torch.float32,
                                    generator=generator)
                if shared is not None and "slot" in command:
                    chunk = shared.read(command["slot"], len(command["batch"]["capture_ms"]))
                else:
                    chunk = command["pose"], command["face"]
                if "condition_job" in command:
                    latents = conditioner.take(command["condition_job"])
                    result, ready = session.push(chunk, noise, display=True, pose_latents=latents)
                else:
                    result, ready = session.push(chunk, noise, display=True)
                if conditioner is None:
                    torch.cuda.synchronize(device)
                else:
                    torch.cuda.current_stream(device).synchronize()
                last_generation_end = time.perf_counter()
                seconds = last_generation_end - start
                busy_seconds += seconds
                generated_frames += result.shape[0]
                encoder.submit(dict(type="generated", worker=kind, job=job, session_id=sid,
                                    batch=command["batch"], seconds=seconds, queue_seconds=queue_seconds,
                                    idle_seconds=idle_seconds, busy_seconds=busy_seconds,
                                    generated_frames=generated_frames,
                                    elapsed_seconds=last_generation_end - generation_started,
                                    generation_started_at=generation_started,
                                    compiled_graphs=graph_count(),
                                    allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                                    peak_gib=torch.cuda.max_memory_allocated(device) / 2**30), result, ready)
                events.put(dict(type="generation_free", worker=kind, job=job, session_id=sid))
        except Exception as exc:
            events.put(dict(type="error", worker=kind, job=job, session_id=sid,
                            error=str(exc) or type(exc).__name__, traceback=traceback.format_exc()))
