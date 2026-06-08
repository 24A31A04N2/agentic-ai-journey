import asyncio
import json
import httpx
import sys

# Ensure UTF-8 output on Windows console for emojis
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Try to import websockets, print warning if missing
try:
    import websockets
except ImportError:
    print("Warning: 'websockets' library is not installed. WebSockets client test will be skipped.")
    print("To install it, run: pip install websockets")
    websockets = None

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000"

async def test_rest_endpoint(client: httpx.AsyncClient):
    """Tests the standard POST query route with Pydantic validation."""
    print("\n--- 1. Testing REST POST /api/query ---")
    payload = {
        "prompt": "Search for latest Agentic AI framework design patterns",
        "session_id": "session-xyz-123",
        "max_steps": 4
    }
    
    try:
        response = await client.post(f"{BASE_URL}/api/query", json=payload)
        if response.status_code == 200:
            print("[REST Success] Response received:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"[REST Failure] Status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[REST Error] Could not connect to server: {e}")

async def test_sse_endpoint(client: httpx.AsyncClient):
    """Tests the SSE streaming route /api/stream."""
    print("\n--- 2. Testing SSE GET /api/stream ---")
    params = {"prompt": "Generate a summary of LLM agents"}
    
    try:
        # We use client.stream to handle SSE chunks
        async with client.stream("GET", f"{BASE_URL}/api/stream", params=params) as response:
            if response.status_code != 200:
                print(f"[SSE Failure] Status code {response.status_code}")
                return
            
            print("[SSE Connection Established] Reading stream chunks...")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    # Extract the JSON payload from the SSE format
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        print(f"[{data['stage']}] Step {data['step_index']}: {data['message']} (Status: {data['status']})")
                    except json.JSONDecodeError:
                        print(f"Raw chunk: {line}")
    except Exception as e:
        print(f"[SSE Error] Connection failed: {e}")

async def test_websocket_endpoint():
    """Tests the WebSockets route /ws/chat/{client_id}."""
    print("\n--- 3. Testing WebSockets WS /ws/chat/{client_id} ---")
    if websockets is None:
        print("[Skipped] websockets package is missing.")
        return
    
    client_id = "User_Battula"
    uri = f"{WS_URL}/ws/chat/{client_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            # 1. Read handshake message
            welcome = await websocket.recv()
            print(f"[WS Server -> Client]: {welcome}")
            
            # 2. Read join announcement
            announcement = await websocket.recv()
            print(f"[WS Broadcast]: {announcement}")
            
            # 3. Send message to the agent
            message_payload = {"message": "Hello Agent! Show me your power!"}
            await websocket.send(json.dumps(message_payload))
            print(f"[WS Client -> Server]: Sent message: {message_payload['message']}")
            
            # 4. Receive user message echo
            echo = await websocket.recv()
            print(f"[WS Broadcast (Echo)]: {echo}")
            
            # 5. Receive agent typing message
            typing = await websocket.recv()
            print(f"[WS Server (Status)]: {typing}")
            
            # 6. Receive agent reply
            reply = await websocket.recv()
            print(f"[WS Server -> Client (Agent Reply)]: {reply}")
            
            # 7. Say goodbye and close
            print("[WS Closing] Chat session finished successfully.")
            
    except Exception as e:
        print(f"[WS Error] Connection failed: {e}")

async def main():
    print("==================================================")
    print("🤖 DAY 6 AGENTIC API CLIENT INITIALIZED")
    print("==================================================")
    
    # We will check if the server is running by doing a quick health-check GET
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{BASE_URL}/")
            print(f"Server is online: {health.json()['message']}")
        except Exception:
            print(f"❌ Error: FastAPI server is not running on {BASE_URL}.")
            print("Please start the server first in another terminal:")
            print("   uvicorn main:app --reload")
            sys.exit(1)
            
        await test_rest_endpoint(client)
        await test_sse_endpoint(client)
        
    await test_websocket_endpoint()
    print("\n==================================================")
    print("✅ Day 6 API Testing Completed.")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
