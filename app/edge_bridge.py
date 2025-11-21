"""Edge bridge for subscribing to MQTT topics and caching latest messages."""

import json
import logging
import threading
import time
from typing import Dict, Optional, Any
from datetime import datetime
import paho.mqtt.client as mqtt
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class EdgeBridge:
    """Thread-safe bridge to edge runtime via MQTT."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize edge bridge.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Thread-safe cache for latest messages
        self._lock = threading.Lock()
        self._latest_state: Optional[Dict[str, Any]] = None
        self._latest_adapt: Optional[Dict[str, Any]] = None
        
        # Offline ring buffer (60 seconds of messages)
        self._ring_buffer: list = []
        self._ring_buffer_max_size = 60  # ~60 seconds at 1 msg/sec
        self._ring_buffer_lock = threading.Lock()
        
        # MQTT client
        self.client: Optional[mqtt.Client] = None
        self._connected = False
        
        # Start MQTT connection in background thread
        self._start_mqtt()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default config
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return {
                'decision': {
                    'source': 'engine',
                    'hybrid_weights': {'engine': 0.5, 'edge': 0.5}
                },
                'topics': {
                    'state_output': 'edon/state',
                    'adapt_output': 'edon/adapt'
                },
                'mqtt': {
                    'broker': 'localhost',
                    'port': 1883
                }
            }
    
    def _start_mqtt(self):
        """Start MQTT client in background thread."""
        def mqtt_thread():
            try:
                mqtt_config = self.config.get('mqtt', {})
                broker = mqtt_config.get('broker', 'localhost')
                port = mqtt_config.get('port', 1883)
                
                self.client = mqtt.Client(client_id=f"edon_bridge_{int(time.time())}")
                self.client.on_connect = self._on_connect
                self.client.on_message = self._on_message
                self.client.on_disconnect = self._on_disconnect
                
                # Connect
                try:
                    self.client.connect(broker, port, 60)
                    self.client.loop_start()
                    logger.info(f"MQTT client connecting to {broker}:{port}")
                except Exception as e:
                    logger.error(f"Failed to connect to MQTT broker: {e}")
                    logger.info("Edge bridge will continue without MQTT (edge mode disabled)")
            except Exception as e:
                logger.error(f"Error starting MQTT client: {e}")
        
        thread = threading.Thread(target=mqtt_thread, daemon=True)
        thread.start()
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            was_connected = self._connected
            self._connected = True
            logger.info("MQTT connected successfully")
            
            # Subscribe to topics
            topics = self.config.get('topics', {})
            state_topic = topics.get('state_output', 'edon/state')
            adapt_topic = topics.get('adapt_output', 'edon/adapt')
            
            client.subscribe(state_topic)
            client.subscribe(adapt_topic)
            logger.info(f"Subscribed to {state_topic} and {adapt_topic}")
            
            # Replay ring buffer on reconnect (if we were disconnected)
            if not was_connected:
                self._replay_ring_buffer()
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            self._connected = False
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            topic = msg.topic
            
            topics = self.config.get('topics', {})
            
            # Add timestamp if not present
            if 'ts' not in payload:
                payload['ts'] = datetime.utcnow().isoformat() + 'Z'
            
            # Store in ring buffer
            with self._ring_buffer_lock:
                self._ring_buffer.append({
                    'topic': topic,
                    'payload': payload,
                    'timestamp': time.time()
                })
                # Keep only last 60 messages (ring buffer)
                if len(self._ring_buffer) > self._ring_buffer_max_size:
                    self._ring_buffer.pop(0)
            
            with self._lock:
                if topic == topics.get('state_output', 'edon/state'):
                    self._latest_state = payload
                elif topic == topics.get('adapt_output', 'edon/adapt'):
                    self._latest_adapt = payload
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self._connected = False
        logger.warning("MQTT disconnected")
    
    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """Get latest state message (thread-safe)."""
        with self._lock:
            return self._latest_state.copy() if self._latest_state else None
    
    def get_latest_adapt(self) -> Optional[Dict[str, Any]]:
        """Get latest adapt message (thread-safe)."""
        with self._lock:
            return self._latest_adapt.copy() if self._latest_adapt else None

    def update_state(self, message: Dict[str, Any]):
        """Update latest state message (thread-safe)."""
        if message is None:
            return
        message.setdefault("schema", "1.0.0")
        message.setdefault("ts", datetime.utcnow().isoformat() + "Z")
        with self._lock:
            self._latest_state = message

    def update_adapt(self, message: Dict[str, Any]):
        """Update latest adapt message (thread-safe)."""
        if message is None:
            return
        message.setdefault("schema", "1.0.0")
        message.setdefault("ts", datetime.utcnow().isoformat() + "Z")
        with self._lock:
            self._latest_adapt = message
    
    def is_connected(self) -> bool:
        """Check if MQTT is connected."""
        return self._connected
    
    def get_decision_config(self) -> Dict:
        """Get decision configuration."""
        return self.config.get('decision', {
            'source': 'engine',
            'hybrid_weights': {'engine': 0.5, 'edge': 0.5}
        })
    
    def _replay_ring_buffer(self):
        """Replay messages from ring buffer after reconnection."""
        with self._ring_buffer_lock:
            if not self._ring_buffer:
                return
            
            logger.info(f"Replaying {len(self._ring_buffer)} messages from ring buffer")
            topics = self.config.get('topics', {})
            
            for msg_entry in self._ring_buffer:
                topic = msg_entry['topic']
                payload = msg_entry['payload']
                
                # Update latest state/adapt from buffer
                with self._lock:
                    if topic == topics.get('state_output', 'edon/state'):
                        self._latest_state = payload
                    elif topic == topics.get('adapt_output', 'edon/adapt'):
                        self._latest_adapt = payload
    
    def get_ring_buffer_size(self) -> int:
        """Get current ring buffer size."""
        with self._ring_buffer_lock:
            return len(self._ring_buffer)
    
    def close(self):
        """Close MQTT connection."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


# Global instance (singleton)
_edge_bridge: Optional[EdgeBridge] = None


def get_edge_bridge() -> EdgeBridge:
    """Get or create global edge bridge instance."""
    global _edge_bridge
    if _edge_bridge is None:
        _edge_bridge = EdgeBridge()
    return _edge_bridge

