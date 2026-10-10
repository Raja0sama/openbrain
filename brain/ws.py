"""Just enough RFC 6455 to push events to a browser and read its messages. Stdlib only."""
import base64
import hashlib
import json
import struct
import threading

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class Closed(Exception):
    pass


class Socket:
    def __init__(self, handler):
        self.r, self.w = handler.rfile, handler.wfile
        self.lock = threading.Lock()
        key = handler.headers["Sec-WebSocket-Key"]
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        handler.send_response(101)
        handler.send_header("Upgrade", "websocket")
        handler.send_header("Connection", "Upgrade")
        handler.send_header("Sec-WebSocket-Accept", accept)
        handler.end_headers()
        handler.wfile.flush()

    def _frame(self, opcode, data):
        head = bytes([0x80 | opcode])
        n = len(data)
        if n < 126:
            head += bytes([n])
        elif n < 65536:
            head += bytes([126]) + struct.pack(">H", n)
        else:
            head += bytes([127]) + struct.pack(">Q", n)
        with self.lock:
            try:
                self.w.write(head + data)
                self.w.flush()
            except OSError as e:
                raise Closed from e

    def send(self, event):
        self._frame(0x1, json.dumps(event, separators=(",", ":")).encode())

    def _read(self, n):
        data = self.r.read(n)
        if len(data) < n:
            raise Closed
        return data

    def recv(self):
        """The next text message, parsed as JSON. Raises Closed when the browser leaves."""
        while True:
            b1, b2 = self._read(2)
            opcode, n = b1 & 0x0F, b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b2 & 0x80 else b"\0\0\0\0"
            data = bytes(c ^ mask[i % 4] for i, c in enumerate(self._read(n)))
            if opcode == 0x8:
                raise Closed
            if opcode == 0x9:
                self._frame(0xA, data)
            elif opcode == 0x1:
                return json.loads(data or b"{}")
