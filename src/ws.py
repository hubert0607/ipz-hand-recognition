import asyncio
import json
import websockets
from recognition import detected_persons

async def send_data(websocket):
    """Send data every 25ms to connected WebSocket clients."""
    data = detected_persons
    try:
        while True:
            # Serialize data to JSON
            message = json.dumps(data)
            await websocket.send(message)
            print(f"Sent: {message}")
            await asyncio.sleep(0.025)  # 25ms interval
    except websockets.exceptions.ConnectionClosedError:
        print("Connection closed by the client.")

async def websocket_handler(websocket, path):
    """Handle incoming WebSocket connections."""
    print(f"Client connected: {path}")
    await send_data(websocket)

async def start_server():
    """Start the WebSocket server locally."""
    server = await websockets.serve(websocket_handler, "127.0.0.1", 8765)
    print("Server running on ws://127.0.0.1:8765")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("Server stopped.")
