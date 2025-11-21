import time
import math
import random
import sys
import os

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sdk', 'python'))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

# Import SDK client (same as test_sdk_quick.py)
from edon_sdk import EdonClient

BASE_URL = "http://127.0.0.1:8001"
WINDOW_LEN = 240

console = Console()
client = EdonClient(base_url=BASE_URL, timeout=5.0, verbose=False)  # Set verbose=False for cleaner output


def make_fake_window(seed=None):
    """Simulate a physiological + environmental data window."""
    # IMPORTANT: make this match whatever test_sdk_quick / load_test uses
    if seed is None:
        seed = random.randint(0, 10000)

    # Add base_stress to make data more dynamic
    base_stress = random.choice([0.0, 0.2, 0.5, 0.9])  # occasionally simulate stressy windows

    return {
        "EDA":       [0.01 * (k % 50) + base_stress * 0.02 for k in range(WINDOW_LEN)],
        "TEMP":      [36.0 + (0.3 * base_stress) for _ in range(WINDOW_LEN)],
        "BVP":       [math.sin((k + seed) / 12.0) * (1 + base_stress) for k in range(WINDOW_LEN)],
        "ACC_x":     [0.0] * WINDOW_LEN,
        "ACC_y":     [0.0] * WINDOW_LEN,
        "ACC_z":     [1.0] * WINDOW_LEN,
        "temp_c":    22.0 + random.uniform(-2, 2) * base_stress,
        "humidity":  50.0 + random.uniform(-5, 5) * base_stress,
        "aqi":       random.choice([20, 35, 60, 80, 120]),
        "local_hour": random.randint(0, 23),
    }


def build_view(result, step, latency_ms):
    """Build a rich layout (panel + table) for the latest EDON result."""
    # SDK client.cav() returns the result directly (already extracted from batch response)
    state = result.get("state", "unknown")
    cav_raw = result.get("cav_raw", 0)
    cav_smooth = result.get("cav_smooth", cav_raw)
    parts = result.get("parts", {})
    
    # Extract component parts
    bio = parts.get("bio", 0.0) if parts else 0.0
    env = parts.get("env", 0.0) if parts else 0.0
    circadian = parts.get("circadian", 0.0) if parts else 0.0
    p_stress = parts.get("p_stress", 0.0) if parts else 0.0

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Step", str(step))
    table.add_row("Last latency", f"{latency_ms:.1f} ms")
    table.add_row("State", f"[bold]{state}[/bold]")
    table.add_row("CAV (raw)", f"{cav_raw:,}")
    table.add_row("CAV (smooth)", f"{cav_smooth:,}")
    table.add_row("", "")  # Spacer
    table.add_row("Bio", f"{bio:.4f}")
    table.add_row("Env", f"{env:.4f}")
    table.add_row("Circadian", f"{circadian:.4f}")
    table.add_row("P(Stress)", f"{p_stress:.4f}")

    panel = Panel(
        table,
        title="[bold cyan]EDON Live Inference[/bold cyan]",
        subtitle="client.cav(window) → /oem/cav/batch",
        border_style="cyan",
    )
    return panel


def main():
    console.print("[bold green]Starting EDON live demo (Ctrl+C to stop)...[/bold green]")
    console.print(f"[dim]Connecting to {BASE_URL} using SDK[/dim]\n")
    
    step = 0
    with Live(console=console, refresh_per_second=4) as live:
        while True:
            try:
                step += 1
                window = make_fake_window()
                # Use SDK here, not raw requests
                start = time.time()
                result = client.cav(window)
                latency_ms = (time.time() - start) * 1000
                panel = build_view(result, step, latency_ms)
                live.update(panel)
                time.sleep(1.5)
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Demo stopped by user.[/bold yellow]")
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                time.sleep(2)


if __name__ == "__main__":
    main()

