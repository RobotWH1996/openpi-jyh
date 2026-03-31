"""
msgpack numpy serialization + WebSocket client for communicating with serve_policy.py.

Protocol matches openpi_client/msgpack_numpy.py.
"""

import functools
import logging
import threading

import numpy as np

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    logging.warning("msgpack not installed. Run: pip3 install msgpack --user")

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("websocket-client not installed. Run: pip3 install websocket-client --user")


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


if MSGPACK_AVAILABLE:
    MsgpackPacker = functools.partial(msgpack.Packer, default=_pack_array)
    msgpack_packb = functools.partial(msgpack.packb, default=_pack_array)
    msgpack_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class CR100WebSocketClient:
    """WebSocket client for communicating with serve_policy.py using msgpack protocol."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.url = f"ws://{host}:{port}"
        self.ws = None
        self.connected = False
        self.lock = threading.Lock()
        self.metadata = None
        self.packer = None

    def connect(self) -> bool:
        if not WEBSOCKET_AVAILABLE:
            logging.error("websocket-client not available")
            return False
        if not MSGPACK_AVAILABLE:
            logging.error("msgpack not available")
            return False

        try:
            self.ws = websocket.create_connection(self.url, timeout=10)
            self.connected = True
            self.packer = MsgpackPacker()
            metadata_bytes = self.ws.recv()
            self.metadata = msgpack_unpackb(metadata_bytes, raw=False)
            logging.info(f"Connected to server. Metadata: {self.metadata}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to {self.url}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.connected = False

    def send_observation(self, observation: dict, **kwargs) -> dict | None:
        """Send observation and receive action using msgpack protocol.

        Returns the full result dict (with 'actions', 'actions_original', etc.)
        or None on failure.
        """
        if not self.connected:
            return None

        if kwargs:
            observation["__rtc_kwargs__"] = kwargs

        with self.lock:
            try:
                data = self.packer.pack(observation)
                self.ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)

                response_bytes = self.ws.recv()

                if isinstance(response_bytes, str):
                    logging.error(f"Server error: {response_bytes}")
                    return None

                result = msgpack_unpackb(response_bytes, raw=False)

                if isinstance(result, dict) and "actions" in result:
                    return result
                elif isinstance(result, np.ndarray):
                    return {"actions": result}
                else:
                    logging.error(f"Unexpected response format: {type(result)}")
                    return None

            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                self.connected = False
                return None
