import torch


def _owned_tensors(module):
    for owner in module.modules():
        for attr, tensor in list(owner._parameters.items()) + list(owner._buffers.items()):
            if tensor is not None and tensor.device.type != "meta":
                yield owner, attr, tensor


class BlockStreamer:
    """Holds transformer blocks in CPU memory and streams them through a few GPU slots.

    Every block keeps the same layout, so the slots are allocated once and reused.
    Copies run on their own stream: block i+1 is fetched while block i computes.
    """

    def __init__(self, blocks, device, slots=3):
        self.blocks = list(blocks)
        self.slots = slots
        self.copy_stream = torch.cuda.Stream(device=device)

        self.entries = []
        for block in self.blocks:
            block_entries = []
            for owner, attr, tensor in _owned_tensors(block):
                host = tensor.detach().to("cpu")
                try:
                    host = host.pin_memory()
                except RuntimeError:
                    pass
                owner._parameters.pop(attr, None)
                owner._buffers.pop(attr, None)
                setattr(owner, attr, host)
                block_entries.append((owner, attr, host))
            self.entries.append(block_entries)

        layout = [[(t.shape, t.dtype) for _, _, t in e] for e in self.entries]
        if any(item != layout[0] for item in layout):
            raise ValueError("block streaming needs every block to share one layout")

        self.buffers = [[torch.empty(t.shape, dtype=t.dtype, device=device)
                         for _, _, t in self.entries[0]] for _ in range(slots)]
        self.ready = [torch.cuda.Event() for _ in range(slots)]
        self.free = [torch.cuda.Event() for _ in range(slots)]
        self.resident = [None] * slots

        for index, block in enumerate(self.blocks):
            block.register_forward_pre_hook(self._on_enter(index))
            block.register_forward_hook(self._on_exit(index))

    def _fetch(self, index):
        if not 0 <= index < len(self.blocks):
            return
        slot = index % self.slots
        if self.resident[slot] == index:
            return
        self.copy_stream.wait_event(self.free[slot])
        with torch.cuda.stream(self.copy_stream):
            for target, (_, _, host) in zip(self.buffers[slot], self.entries[index]):
                target.copy_(host, non_blocking=True)
        self.ready[slot].record(self.copy_stream)
        self.resident[slot] = index

    def _on_enter(self, index):
        def hook(module, args):
            slot = index % self.slots
            self._fetch(index)
            torch.cuda.current_stream().wait_event(self.ready[slot])
            for (owner, attr, _), buffer in zip(self.entries[index], self.buffers[slot]):
                setattr(owner, attr, buffer)
            self._fetch(index + 1)
        return hook

    def _on_exit(self, index):
        def hook(module, args, output):
            self.free[index % self.slots].record()
            for owner, attr, host in self.entries[index]:
                setattr(owner, attr, host)
        return hook


class KVCacheStreamer:
    """Keeps the rolling KV cache in pinned CPU memory, one layer on the GPU at a time.

    Written back only on the step that commits it, and the window trim runs on the
    GPU so the host copy already has its final size.
    """

    def __init__(self, blocks, device):
        self.device = device
        self.cache = None
        self.host = {}
        self.staged = [None] * len(blocks)
        self.frame_seq_length = None
        self.window = None
        for index, block in enumerate(blocks):
            block.register_forward_pre_hook(self._on_enter(index))
            block.register_forward_hook(self._on_exit(index))

    def bind(self, cache, frame_seq_length, window):
        self.cache = cache
        self.frame_seq_length = frame_seq_length
        self.window = window

    def _host_view(self, index, key, tensor):
        buffer = self.host.get((index, key))
        if buffer is None or buffer.shape[1] < tensor.shape[1]:
            shape = (tensor.shape[0], tensor.shape[1]) + tuple(tensor.shape[2:])
            try:
                buffer = torch.empty(shape, dtype=tensor.dtype, pin_memory=True)
            except RuntimeError:
                buffer = torch.empty(shape, dtype=tensor.dtype)
            self.host[(index, key)] = buffer
        view = buffer[:, :tensor.shape[1]]
        view.copy_(tensor, non_blocking=True)
        return view

    def _trim(self, tensor):
        head = self.frame_seq_length * 4
        tail = self.frame_seq_length * 3
        return torch.cat([tensor[:, :head], tensor[:, -tail:]], dim=1)

    def _on_enter(self, index):
        def hook(module, args):
            if self.cache is None:
                return
            entry = self.cache[index]
            host = {key: entry[key] for key in ("k", "v")}
            uploaded = {key: host[key].to(self.device, non_blocking=True) for key in ("k", "v")}
            entry.update(uploaded)
            self.staged[index] = (host, uploaded)
        return hook

    def _on_exit(self, index):
        def hook(module, args, output):
            if self.staged[index] is None:
                return
            entry = self.cache[index]
            host, uploaded = self.staged[index]
            trim = entry["shift"] > self.window
            for key in ("k", "v"):
                if entry[key] is uploaded[key]:
                    entry[key] = host[key]
                    continue
                tensor = self._trim(entry[key]) if trim else entry[key]
                entry[key] = self._host_view(index, key, tensor)
            if trim:
                entry["shift"] = self.window
            self.staged[index] = None
        return hook
