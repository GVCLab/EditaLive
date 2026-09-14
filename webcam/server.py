import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT
from .protocol import unpack
from .service import WebcamService


def create_app(paths=None, service=None):
    service = service or WebcamService(paths)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await service.close()

    app = FastAPI(lifespan=lifespan)
    app.state.service = service

    @app.get("/api/status")
    async def status():
        return service.status()

    @app.websocket("/api/ws")
    async def websocket(socket: WebSocket):
        await socket.accept()
        if service.socket is not None:
            await socket.send_json(dict(type="error", message="Another page is connected. Close it before reconnecting."))
            await socket.close(code=1013)
            return
        service.socket = socket
        await service.send(service.status())
        try:
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                try:
                    if message.get("bytes") is not None:
                        header, payload = unpack(message["bytes"])
                        action = header.get("type")
                        if action == "frame":
                            await service.receive_frame(header, payload)
                        elif action == "start":
                            service.launch(service.start, header, payload)
                        elif action == "reset":
                            service.launch(service.reset, payload)
                        else:
                            raise ValueError("Unknown image message.")
                    elif message.get("text") is not None:
                        import json
                        data = json.loads(message["text"])
                        if not isinstance(data, dict):
                            raise ValueError("Control messages must be objects.")
                        action = data.get("type")
                        if action == "prepare_model":
                            service.launch(service.prepare_model, data.get("config", {}))
                        elif action == "prepare_prompt":
                            service.launch(service.prepare_prompt, data.get("prompt", ""), data.get("config", {}))
                        elif action == "prepare":
                            service.launch(service.prepare, data.get("config", {}), data.get("prompt", ""))
                        elif action == "stop":
                            service.launch(service.stop)
                        elif action == "reset":
                            service.launch(service.reset)
                        elif action == "fps":
                            await service.set_fps(data.get("fps"))
                        elif action == "played":
                            await service.played(data)
                        elif action == "status":
                            await service.send(service.status())
                        else:
                            raise ValueError("Unknown control message.")
                except (ValueError, TypeError, KeyError) as exc:
                    await service.send(dict(type="error", message=str(exc)))
        except WebSocketDisconnect:
            pass
        finally:
            await service.disconnect(socket)

    static = ROOT / "webcam/frontend/dist"
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    else:
        @app.get("/")
        async def missing_frontend():
            return HTMLResponse("<p>Build the frontend: cd webcam/frontend &amp;&amp; npm install &amp;&amp; npm run build</p>")
    return app
