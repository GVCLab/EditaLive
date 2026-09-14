import asyncio
import logging
import math
import multiprocessing as mp
import queue
import secrets
import time
import uuid

from starlette.websockets import WebSocketDisconnect

from .config import ModelPaths, RuntimeConfig
from .flow import CHUNK, FIRST_CHUNK, OUTPUT_LEAD, CompletionRate, FrameBuffer
from .images import crop_reference
from .protocol import pack
from .workers import worker
from .shared import SharedFrames

log = logging.getLogger(__name__)


class WebcamService:
    def __init__(self, paths=None):
        self.paths = paths or ModelPaths()
        self.config = None
        self.state = "idle"
        self.message = "Choose your settings and prepare"
        self.socket = None
        self.send_lock = asyncio.Lock()
        self.processes = {}
        self.commands = {}
        self.events = None
        self.shared = None
        self.pump_task = None
        self.operation = None
        self.pending = {}
        self.job_id = 0
        self.prepared = False
        self.prepared_prompt = None
        self.preparation_started = None
        self.preparation_workers = {}
        self.session_id = None
        self.settings = None
        self.reference = None
        self.metrics = {}
        self.reset_count = 0
        self.driving_fps = 15.0
        self._reset_flow()

    def _reset_flow(self):
        self.buffer = FrameBuffer()
        self.input_credit = 0
        self.output_reserved = 0
        self.output_pending = set()
        self.output_id = 0
        self.next_chunk = 0
        self.extracting = None
        self.chunk_seconds = None
        self.generating = None
        self.prefetched = None
        self.conditioning = None
        self.free_slots = set(range(SharedFrames.slots))
        self.generation_slots = {}
        self.completion_rate = CompletionRate()

    def status(self):
        preparation = None
        if self.state == "preparing" and not self.prepared and self.preparation_started is not None:
            preparation = dict(elapsed_seconds=time.perf_counter() - self.preparation_started,
                               workers={kind: dict(progress) for kind, progress in self.preparation_workers.items()})
        return dict(type="status", state=self.state, message=self.message,
                    prepared=self.prepared, prepared_prompt=self.prepared_prompt, config=self.config.model_dump() if self.config else None,
                    session_id=self.session_id, driving_fps=self.driving_fps, reset_count=self.reset_count,
                    input_frames=len(self.buffer.frames), output_frames=self.output_reserved,
                    preparation=preparation, **self.metrics)

    async def send(self, message, payload=None):
        socket = self.socket
        if socket is None:
            return
        async with self.send_lock:
            if self.socket is not socket:
                return
            try:
                if payload is None:
                    await socket.send_json(message)
                else:
                    await socket.send_bytes(pack(message, payload))
            except (WebSocketDisconnect, OSError):
                pass

    async def set_state(self, state, message):
        self.state, self.message = state, message
        await self.send(self.status())

    def launch(self, method, *args):
        if self.operation is not None and not self.operation.done():
            raise ValueError("Wait for the current operation to finish.")
        self.operation = asyncio.create_task(self._run(method, args))

    async def _run(self, method, args):
        try:
            await method(*args)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Webcam operation failed")
            message = str(exc) or type(exc).__name__
            if self.state in {"preparing", "starting", "stopping"}:
                self.session_id = None
                await self.set_state("error", message)
            await self.send(dict(type="error", message=message))

    def _put(self, kind, action, **fields):
        self.job_id += 1
        job = self.job_id
        self.commands[kind].put_nowait(dict(type=action, job=job, queued_at=time.perf_counter(), **fields))
        return job

    async def _rpc(self, kind, action, **fields):
        job = self._put(kind, action, **fields)
        future = asyncio.get_running_loop().create_future()
        self.pending[job] = future
        try:
            return await asyncio.wait_for(future, timeout=1800)
        finally:
            self.pending.pop(job, None)

    def _check_preparation(self):
        if self.state in {"running", "starting", "stopping", "preparing"}:
            raise ValueError("Stop the current session before preparing.")

    async def prepare_model(self, data):
        self._check_preparation()
        config = RuntimeConfig(**data)
        if self.prepared and config == self.config:
            await self.set_state("ready", "Ready" if self.prepared_prompt else "Model ready. Prepare a prompt.")
            return
        self.paths.validate_files(config)
        self.preparation_started = time.perf_counter()
        self.preparation_workers = {kind: dict(message="Starting worker", done=False)
                                    for kind in ("generation", "extraction")}
        await self._shutdown_workers()
        self.config = config
        self.metrics = {}
        self._reset_flow()
        await self.set_state("preparing", "Loading models")
        try:
            ctx = mp.get_context("spawn")
            self.events = ctx.Queue(maxsize=8)
            self.shared = SharedFrames(ctx, config.size)
            condition_commands = ctx.Queue(maxsize=2) if config.dual_gpu else None
            if condition_commands is not None:
                self.commands["condition"] = condition_commands
            self.ready_workers = set()
            self.ready_future = asyncio.get_running_loop().create_future()
            for kind in ("generation", "extraction"):
                commands = ctx.Queue(maxsize=2)
                process = ctx.Process(target=worker, args=(kind, config.model_dump(), self.paths.model_dump(),
                                                           commands, self.events, self.shared,
                                                           condition_commands if kind == "generation" else None), daemon=True)
                try:
                    process.start()
                except BaseException:
                    commands.close()
                    raise
                self.commands[kind], self.processes[kind] = commands, process
            self.pump_task = asyncio.create_task(self._pump())
            await asyncio.wait_for(self.ready_future, timeout=1800)
            self.prepared = True
            await self.set_state("ready", "Model ready. Prepare a prompt.")
        except BaseException:
            await self._shutdown_workers()
            await self.set_state("error", "Model preparation failed")
            raise

    async def prepare_prompt(self, prompt, data=None):
        self._check_preparation()
        if not self.prepared or (data is not None and RuntimeConfig(**data) != self.config):
            raise ValueError("Prepare the selected model settings first.")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 4000:
            raise ValueError("Enter a prompt of 1 to 4000 characters.")
        self.prepared_prompt = None
        await self.set_state("preparing", "Preparing prompt")
        try:
            await self._rpc("generation", "prepare", prompt=prompt.strip())
            self.prepared_prompt = prompt.strip()
            await self.set_state("ready", "Ready")
        except BaseException:
            await self.set_state("error", "Prompt preparation failed")
            raise

    async def prepare(self, data, prompt):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 4000:
            raise ValueError("Enter a prompt of 1 to 4000 characters.")
        await self.prepare_model(data)
        await self.prepare_prompt(prompt, data)

    async def start(self, header, payload):
        if self.state != "ready" or not self.prepared:
            raise ValueError("Prepare the model and prompt first.")
        prompt = header.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 4000:
            raise ValueError("Enter a prompt of 1 to 4000 characters.")
        prompt = prompt.strip()
        if prompt != self.prepared_prompt:
            raise ValueError("Prepare this prompt before starting.")
        mode = header.get("reference_mode", "camera")
        if mode != "camera":
            raise ValueError("Only camera reference frames are supported in v1.")
        crop = header.get("crop")
        reference = await asyncio.to_thread(crop_reference, payload, crop, self.config.size)
        seed = header.get("seed", -1)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < -1 or seed >= 2**63:
            raise ValueError("Invalid seed.")
        self.settings = dict(prompt=prompt, reference_mode=mode, crop=crop, seed=seed)
        self.reference = reference
        await self._start_session()

    async def _start_session(self):
        self._reset_flow()
        self.metrics = {}
        self.session_id = uuid.uuid4().hex
        await self.set_state("starting", "Preparing camera reference")
        seed = self.settings["seed"]
        if seed < 0:
            seed = secrets.randbelow(2**31)
        try:
            await self._rpc("generation", "start", session_id=self.session_id,
                            reference=self.reference, prompt=self.settings["prompt"], seed=seed)
        except Exception:
            self.session_id = None
            await self.set_state("error", "Session initialization failed")
            raise
        await self.set_state("running", "Live")
        await self._grant_credits()

    async def stop(self):
        if self.state == "preparing":
            raise ValueError("Wait for preparation to finish.")
        if not self.prepared:
            return
        self.session_id = None
        await self.set_state("stopping", "Stopping")
        await asyncio.gather(self._rpc("generation", "close"), self._rpc("extraction", "close"))
        self._reset_flow()
        await self.set_state("ready", "Ready")

    async def reset(self, payload=None):
        if self.state not in {"running", "error"} or self.settings is None:
            raise ValueError("There is no session to restart.")
        if not payload:
            raise ValueError("Restart requires a fresh camera reference frame.")
        reference = await asyncio.to_thread(crop_reference, payload, self.settings["crop"], self.config.size)
        await self.stop()
        self.reference = reference
        self.reset_count += 1
        await self._start_session()

    async def set_fps(self, fps):
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 8 <= fps <= 25:
            raise ValueError("Capture FPS must be between 8 and 25.")
        self.driving_fps = float(fps)
        await self.send(self.status())

    async def receive_frame(self, header, payload):
        if self.state != "running" or header.get("session_id") != self.session_id:
            return
        if self.input_credit <= 0:
            raise ValueError("No input capacity is available.")
        self.buffer.append(header, payload)
        self.input_credit -= 1
        await self._schedule()

    async def played(self, data):
        if data.get("session_id") != self.session_id:
            return
        frame_id = data.get("output_id")
        if frame_id in self.output_pending:
            self.output_pending.remove(frame_id)
            self.output_reserved -= 1
            await self._schedule()

    async def _grant_credits(self):
        if self.state != "running":
            return
        credit = self.buffer.capacity - len(self.buffer.frames) - self.input_credit
        if credit > 0:
            self.input_credit += credit
            await self.send(dict(type="credit", session_id=self.session_id, count=credit,
                                 driving_fps=self.driving_fps))

    async def _schedule(self):
        if self.state != "running":
            return
        if self.generating is None and self.prefetched is not None:
            event = self.prefetched
            count = len(event["batch"]["capture_ms"])
            if self.output_reserved + count <= OUTPUT_LEAD * CHUNK:
                self.prefetched = None
                self.output_reserved += count
                payload = ({"slot": event["slot"]} if "slot" in event
                           else {"pose": event["pose"], "face": event["face"]})
                if event.get("type") == "conditioned":
                    payload["condition_job"] = event["job"]
                self.generating = self._put("generation", "generate", session_id=self.session_id,
                                            batch=event["batch"], **payload)
                if "slot" in event:
                    self.generation_slots[self.generating] = event["slot"]
        if (self.extracting is None and self.prefetched is None and self.conditioning is None
                and (self.shared is None or self.free_slots)):
            count = FIRST_CHUNK if self.next_chunk == 0 else CHUNK
            window = count if self.chunk_seconds is None else math.ceil(self.driving_fps * self.chunk_seconds)
            batch = self.buffer.take(count, self.driving_fps, window)
            if batch is not None:
                batch["chunk_id"] = self.next_chunk
                self.next_chunk += 1
                payload = {"slot": self.free_slots.pop()} if self.shared is not None else {}
                self.extracting = self._put("extraction", "extract", session_id=self.session_id,
                                            batch=batch, **payload)
                await self._grant_credits()

    async def _on_event(self, event):
        kind, job = event.get("type"), event.get("job")
        if kind in {"fatal", "error"}:
            log.error("%s: %s\n%s", event.get("worker"), event["error"], event.get("traceback", ""))
            error = RuntimeError(event["error"])
            if job in self.pending:
                if not self.pending[job].done():
                    self.pending[job].set_exception(error)
            elif self.state == "preparing" and not self.ready_future.done():
                self.ready_future.set_exception(error)
            elif event.get("session_id") == self.session_id or kind == "fatal":
                self.session_id = None
                self._reset_flow()
                await self.set_state("error", event["error"])
            return
        if job in self.pending:
            if not self.pending[job].done():
                self.pending[job].set_result(event)
            return
        if kind == "progress":
            if self.state == "preparing":
                if not self.prepared:
                    self.preparation_workers[event["worker"]] = dict(message=event["message"], done=False)
                await self.set_state("preparing", event["message"])
        elif kind == "ready":
            self.preparation_workers[event["worker"]] = dict(message="Ready", done=True)
            await self.send(self.status())
            self.ready_workers.add(event["worker"])
            if len(self.ready_workers) == 2 and not self.ready_future.done():
                self.ready_future.set_result(True)
        elif kind == "extracted":
            if job == self.extracting:
                self.extracting = None
            if self.state == "running" and event.get("session_id") == self.session_id:
                if self.config.dual_gpu:
                    self.conditioning = self._put("condition", "condition", session_id=self.session_id,
                                                  slot=event["slot"], batch=event["batch"])
                else:
                    self.prefetched = event
                await self._schedule()
        elif kind == "conditioned":
            if job == self.conditioning:
                self.conditioning = None
            if self.state == "running" and event.get("session_id") == self.session_id:
                self.prefetched = event
                await self._schedule()
        elif kind == "generation_free":
            slot = self.generation_slots.pop(job, None)
            if slot is not None:
                self.free_slots.add(slot)
            if job == self.generating:
                self.generating = None
                if self.state == "running" and event.get("session_id") == self.session_id:
                    await self._schedule()
        elif kind == "generated":
            slot = self.generation_slots.pop(job, None)
            if slot is not None:
                self.free_slots.add(slot)
            if job == self.generating:
                self.generating = None
            if self.state != "running" or event.get("session_id") != self.session_id:
                return
            batch = event["batch"]
            sid = self.session_id
            await self._schedule()
            for i, frame in enumerate(event["frames"]):
                if self.session_id != sid:
                    break
                output_id = self.output_id
                self.output_id += 1
                self.output_pending.add(output_id)
                await self.send(dict(type="frame", session_id=sid, output_id=output_id,
                                     chunk_id=batch["chunk_id"], source_frame_id=batch["frame_ids"][i],
                                     capture_ms=batch["capture_ms"][i], chunk_frame=i,
                                     chunk_frames=len(event["frames"]), generation_seconds=event["seconds"],
                                     ratio=batch["ratio"], consumed=batch["consumed"],
                                     source_start=batch["source_start"], source_end=batch["source_end"]), frame)
            if self.session_id == sid:
                self.chunk_seconds = (event["seconds"] if self.chunk_seconds is None
                                      else self.chunk_seconds + 0.3 * (event["seconds"] - self.chunk_seconds))
                fps = self.completion_rate.update(len(event["frames"]),
                                                  event.get("completed_at", time.perf_counter()),
                                                  event.get("generation_started_at"))
                self.metrics = dict(generation_fps=fps,
                                    catchup_ratio=batch["ratio"], gpu_gib=event["allocated_gib"],
                                    peak_gib=event["peak_gib"], compiled_graphs=event.get("compiled_graphs", 0),
                                    generation_seconds=event["seconds"],
                                    generation_idle_seconds=event.get("idle_seconds", 0),
                                    generated_frames=event.get("generated_frames", 0),
                                    generation_busy_seconds=event.get("busy_seconds", 0),
                                    generation_elapsed_seconds=event.get("elapsed_seconds", 0),
                                    extraction_seconds=batch.get("extract_seconds", 0),
                                    queue_seconds=event.get("queue_seconds", 0) + batch.get("queue_seconds", 0))
                await self.send(self.status())
                await self._schedule()

    async def _pump(self):
        def receive():
            try:
                return self.events.get(timeout=0.2)
            except queue.Empty:
                return None
        try:
            while True:
                event = await asyncio.to_thread(receive)
                if event is not None:
                    await self._on_event(event)
                else:
                    for kind, process in self.processes.items():
                        if process.exitcode is not None:
                            error = RuntimeError(f"{kind} worker exited ({process.exitcode})")
                            for future in self.pending.values():
                                if not future.done():
                                    future.set_exception(error)
                            if self.state == "preparing" and not self.ready_future.done():
                                self.ready_future.set_exception(error)
                            else:
                                self.prepared = False
                                self.session_id = None
                                await self.set_state("error", str(error))
                            return
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Worker event loop failed")
            self.prepared = False
            self.session_id = None
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("Worker communication failed"))
            await self.set_state("error", "Worker communication failed")

    async def _shutdown_workers(self):
        self.prepared = False
        self.prepared_prompt = None
        self.session_id = None
        if self.pump_task is not None:
            self.pump_task.cancel()
            await asyncio.gather(self.pump_task, return_exceptions=True)
            self.pump_task = None
        for kind, process in self.processes.items():
            if process.is_alive():
                try:
                    self._put(kind, "shutdown")
                except queue.Full:
                    process.terminate()
        for process in self.processes.values():
            await asyncio.to_thread(process.join, 5)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5)
        for q in [*self.commands.values(), self.events]:
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self.processes, self.commands = {}, {}
        self.events = None
        self.shared = None
        for future in self.pending.values():
            if not future.done():
                future.cancel()
        self.pending.clear()

    async def disconnect(self, socket):
        if self.socket is not socket:
            return
        self.socket = None
        if self.operation is not None and not self.operation.done():
            self.operation.cancel()
            await asyncio.gather(self.operation, return_exceptions=True)
        if self.prepared and self.state != "preparing":
            await self.stop()

    async def close(self):
        if self.operation is not None and not self.operation.done():
            self.operation.cancel()
            await asyncio.gather(self.operation, return_exceptions=True)
        await self._shutdown_workers()
