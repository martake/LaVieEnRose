"""
FastAPI WebSocket server for real-time simulation visualization.
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from simulation import Simulation
from agent_finite_diff import AgentFiniteDiff
from agent_adjoint import AgentAdjoint

app = FastAPI(title="La Vie En Rose — Adjoint Method Comparison")

STATIC_DIR = Path(__file__).parent / "static"

# --- Simulation state ---
sim: Simulation | None = None
running = False
speed_ms = 400  # milliseconds between steps
adjoint_window = 10


def create_simulation(seed: int = 42, window_size: int = 10) -> Simulation:
    agent_a = AgentFiniteDiff(lr=0.02)
    agent_b = AgentAdjoint(lr=0.02, window_size=window_size)
    return Simulation(agent_a, agent_b, seed=seed)


# --- WebSocket connections ---
connections: list[WebSocket] = []


async def broadcast(message: dict):
    """Send message to all connected WebSocket clients."""
    data = json.dumps(message)
    disconnected = []
    for ws in connections:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        connections.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global sim, running, speed_ms, adjoint_window

    await ws.accept()
    connections.append(ws)

    # Initialize simulation if needed
    if sim is None:
        sim = create_simulation(window_size=adjoint_window)

    # Send initial state
    await ws.send_text(json.dumps({
        'type': 'init',
        'running': running,
        'speed_ms': speed_ms,
        'adjoint_window': adjoint_window,
        'step': sim.step_count,
    }))

    try:
        while True:
            msg = await ws.receive_text()
            cmd = json.loads(msg)

            if cmd['action'] == 'start':
                running = True
                await broadcast({'type': 'status', 'running': True})
                asyncio.create_task(run_loop())

            elif cmd['action'] == 'stop':
                running = False
                await broadcast({'type': 'status', 'running': False})

            elif cmd['action'] == 'step':
                if sim is not None:
                    result = sim.step()
                    await broadcast({'type': 'step', **result})

            elif cmd['action'] == 'reset':
                running = False
                window_size = cmd.get('window_size', adjoint_window)
                seed = cmd.get('seed', 42)
                adjoint_window = window_size
                sim = create_simulation(seed=seed, window_size=window_size)
                await broadcast({
                    'type': 'reset',
                    'running': False,
                    'step': 0,
                    'adjoint_window': adjoint_window,
                })

            elif cmd['action'] == 'set_speed':
                speed_ms = max(50, min(2000, cmd.get('speed_ms', 400)))
                await broadcast({'type': 'speed', 'speed_ms': speed_ms})

            elif cmd['action'] == 'set_window':
                adjoint_window = max(2, min(50, cmd.get('window_size', 10)))
                if sim is not None and hasattr(sim.agent_b, 'window_size'):
                    sim.agent_b.window_size = adjoint_window
                await broadcast({'type': 'window', 'adjoint_window': adjoint_window})

    except WebSocketDisconnect:
        connections.remove(ws)


async def run_loop():
    """Continuous simulation loop."""
    global running, sim
    while running and sim is not None:
        result = sim.step()
        await broadcast({'type': 'step', **result})
        await asyncio.sleep(speed_ms / 1000.0)


# --- Static files ---
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
