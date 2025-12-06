"""FastAPI server for demo UI."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import asyncio
from typing import Dict, Any, Optional
import os
import numpy as np


app = FastAPI(title="EDON MuJoCo Demo")

# Global state for demo
demo_state: Dict[str, Any] = {
    "running": False,
    "mode": "baseline",  # "baseline" or "edon"
    "edon_mode": "zero-shot",  # "zero-shot" or "trained"
    "trained_model_path": None,  # Path to trained model if mode is "trained"
    "baseline_metrics": {},
    "edon_metrics": {},
    "baseline_state": {},
    "edon_state": {},
    "edon_enabled": True,
    "kill_switch": False,
}

# WebSocket connections
connections: list = []


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve main HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>EDON MuJoCo Demo</h1><p>UI file not found</p>")


@app.get("/api/state")
async def get_state():
    """Get current demo state."""
    return demo_state


@app.post("/api/control")
async def control(command: Dict[str, Any]):
    """Control demo (start, stop, toggle EDON, kill switch)."""
    cmd = command.get("command")
    
    print(f"[UI] Received command: {cmd}")
    
    if cmd == "start":
        demo_state["running"] = True
        demo_state["kill_switch"] = False
        print(f"[UI] Set demo_state['running'] = True (start command)")
    elif cmd == "stop":
        demo_state["running"] = False
        print(f"[UI] Set demo_state['running'] = False (stop command)")
        # Note: The demo runner will check demo_state["running"] in the simulation loops
        # and set self.running = False accordingly
    elif cmd == "toggle_edon":
        demo_state["edon_enabled"] = not demo_state["edon_enabled"]
    elif cmd == "toggle_mode":
        # Toggle between zero-shot and trained
        if demo_state.get("edon_mode") == "zero-shot":
            demo_state["edon_mode"] = "trained"
            # Default trained model path (can be changed)
            if not demo_state.get("trained_model_path"):
                demo_state["trained_model_path"] = "models/edon_v8_mujoco.pt"
        else:
            demo_state["edon_mode"] = "zero-shot"
            demo_state["trained_model_path"] = None
    elif cmd == "kill_switch":
        demo_state["kill_switch"] = True
        demo_state["running"] = False
        print(f"[UI] Kill switch activated - set demo_state['running'] = False")
    
    # Broadcast to all WebSocket connections
    await broadcast_state()
    
    return {"status": "ok", "state": demo_state}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    connections.append(websocket)
    
    # Set event loop for update_demo_state
    set_event_loop(asyncio.get_event_loop())
    
    try:
        # Send initial state
        await websocket.send_json(demo_state)
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(0.05)  # Update every 50ms (20 FPS)
            try:
                await websocket.send_json(demo_state)
            except WebSocketDisconnect:
                break
            except Exception:
                break
    except WebSocketDisconnect:
        if websocket in connections:
            connections.remove(websocket)


async def broadcast_state():
    """Broadcast state to all WebSocket connections."""
    if not connections:
        return
    
    disconnected = []
    for connection in connections:
        try:
            await connection.send_json(demo_state)
        except Exception as e:
            # Connection closed or error
            disconnected.append(connection)
    
    # Remove disconnected connections
    for conn in disconnected:
        if conn in connections:
            connections.remove(conn)


# Event loop for async operations
_loop: Optional[asyncio.AbstractEventLoop] = None

def set_event_loop(loop: asyncio.AbstractEventLoop):
    """Set the event loop for async operations."""
    global _loop
    _loop = loop

def _make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable types to JSON-compatible types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def update_demo_state(
    mode: str,
    metrics: Dict[str, Any],
    state: Dict[str, Any],
    edon_info: Optional[Dict[str, Any]] = None
):
    """Update demo state (called from main loop)."""
    # Convert numpy arrays to lists for JSON serialization
    metrics_serializable = _make_json_serializable(metrics)
    state_serializable = _make_json_serializable(state)
    edon_info_serializable = _make_json_serializable(edon_info) if edon_info else None
    
    # Ensure step field is included (it should be in state, but double-check)
    if mode == "baseline":
        demo_state["baseline_metrics"] = metrics_serializable
        demo_state["baseline_state"] = state_serializable
        # Debug: Log step count occasionally
        if state_serializable.get("step", 0) % 100 == 0:
            print(f"  [UI] Baseline step update: {state_serializable.get('step', 'N/A')}")
    elif mode == "edon":
        demo_state["edon_metrics"] = metrics_serializable
        demo_state["edon_state"] = state_serializable
        if edon_info_serializable:
            demo_state["edon_info"] = edon_info_serializable
        # Debug: Log step count occasionally
        if state_serializable.get("step", 0) % 100 == 0:
            print(f"  [UI] EDON step update: {state_serializable.get('step', 'N/A')}")
    
    # Always keep running state True when updating (simulation is active)
    # Only set to False when explicitly stopped or comparison completes
    if not demo_state.get("kill_switch", False):
        demo_state["running"] = True
    
    # Trigger broadcast (non-blocking)
    global _loop
    if _loop is not None and _loop.is_running():
        try:
            # Use asyncio.run_coroutine_threadsafe for thread-safe async calls
            future = asyncio.run_coroutine_threadsafe(broadcast_state(), _loop)
            # Don't wait for it, just fire and forget
        except Exception as e:
            # Silently fail - state is still updated
            pass

