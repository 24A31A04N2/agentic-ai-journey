import asyncio
import json
import time
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Generator, AsyncGenerator, List, Dict

app = FastAPI(
    title="🤖 Agentic AI API Gateway",
    description="FastAPI service exposing Agent capabilities via REST, Server-Sent Events (SSE), and WebSockets",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 📋 PYDANTIC SCHEMAS (Validation Layer)
# ==========================================

class QueryInput(BaseModel):
    prompt: str = Field(..., min_length=3, description="The query/instruction for the agent")
    session_id: str = Field(default="session_default", description="Unique identifier for chat context")
    max_steps: int = Field(default=5, ge=1, le=10, description="Max reasoning loops")

class QueryResponse(BaseModel):
    session_id: str
    status: str
    response: str
    steps_executed: int
    execution_time_ms: float

class ChatMessage(BaseModel):
    sender: str
    message: str
    timestamp: float

# ==========================================
# 🧠 AGENT SERVICE (Business Logic Layer)
# ==========================================

class AgentOrchestrator:
    """Simulates an Agent's reasoning loop, database searches, and response generation."""
    
    async def run_reasoning_loop(self, prompt: str, max_steps: int) -> Dict[str, any]:
        start_time = time.time()
        # Mock execution steps
        await asyncio.sleep(0.8)  # Mimic reasoning / planning
        steps = max_steps
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "status": "completed",
            "response": f"Successfully processed query: '{prompt}'. Reasoned through {steps} steps and resolved target.",
            "steps_executed": steps,
            "execution_time_ms": round(execution_time, 2)
        }

    async def run_streaming_reasoning(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generates Server-Sent Events (SSE) tracing the agent's step-by-step thinking."""
        steps = [
            {"step": "Planning", "details": f"Deconstructing prompt: '{prompt}' into tasks."},
            {"step": "Tool Call", "details": "Querying vector database for matching concepts..."},
            {"step": "Verification", "details": "Validating search results against ground-truth indices..."},
            {"step": "Synthesis", "details": "Formulating response structure and formatting code..."},
            {"step": "Final Answer", "details": "Done. Streaming final response to client."}
        ]
        
        for i, step in enumerate(steps, 1):
            await asyncio.sleep(0.6)  # Simulate agent thinking time per step
            payload = {
                "step_index": i,
                "status": "running" if i < len(steps) else "completed",
                "stage": step["step"],
                "message": step["details"]
            }
            # SSE protocol format: "data: <json>\n\n"
            yield f"data: {json.dumps(payload)}\n\n"

# Dependency Injection function
def get_agent_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator()

# ==========================================
# 🚀 API ENDPOINTS (Routing Layer)
# ==========================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Welcome to Day 6 Agentic API Gateway. FastAPI is ready!",
        "endpoints": ["POST /api/query", "GET /api/stream", "WS /ws/chat/{client_id}"]
    }

# 1. Standard POST route with Pydantic validation & Dependency Injection
@app.post("/api/query", response_model=QueryResponse)
async def query_agent(
    payload: QueryInput, 
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """Handles synchronous agent execution requests."""
    try:
        result = await orchestrator.run_reasoning_loop(payload.prompt, payload.max_steps)
        return QueryResponse(
            session_id=payload.session_id,
            status=result["status"],
            response=result["response"],
            steps_executed=result["steps_executed"],
            execution_time_ms=result["execution_time_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Server-Sent Events (SSE) Streaming response
@app.get("/api/stream")
async def stream_agent(
    prompt: str = Query(..., min_length=3),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """Streams the agent's reasoning process step-by-step using Server-Sent Events (SSE)."""
    return StreamingResponse(
        orchestrator.run_streaming_reasoning(prompt),
        media_type="text/event-stream"
    )

# ==========================================
# 🔌 WEBSOCKET ROUTING (Real-Time Bidirectional)
# ==========================================

class ConnectionManager:
    """Manages active WebSockets connections, broadcasting, and private messaging."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/chat/{client_id}")
async def websocket_chat_endpoint(websocket: WebSocket, client_id: str):
    """Bi-directional live chat endpoint for interactive agent sessions."""
    await manager.connect(websocket)
    
    # Send welcome event
    await manager.send_personal_message({
        "event": "handshake",
        "sender": "Agent-System",
        "message": f"Connected client: {client_id}. Type your message to start agent chat session."
    }, websocket)
    
    # Broadcast join notice
    await manager.broadcast({
        "event": "system",
        "sender": "System",
        "message": f"Client {client_id} joined the chat session."
    })
    
    try:
        while True:
            # Wait for client input message
            data = await websocket.receive_text()
            
            try:
                # Attempt to parse as JSON, default to text
                parsed_data = json.loads(data)
                user_msg = parsed_data.get("message", data)
            except json.JSONDecodeError:
                user_msg = data
            
            # Broadcast user message
            await manager.broadcast({
                "event": "message",
                "sender": client_id,
                "message": user_msg
            })
            
            # Simulate real-time agent typing/thinking delay
            await manager.send_personal_message({
                "event": "typing",
                "sender": "Agent",
                "message": "Agent is reasoning..."
            }, websocket)
            await asyncio.sleep(0.5)
            
            # Send agent's reply
            agent_reply = f"🤖 Agent response to '{user_msg}': I analyzed your message and confirmed WebSockets are functioning perfectly!"
            await manager.broadcast({
                "event": "message",
                "sender": "Agent",
                "message": agent_reply
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast({
            "event": "system",
            "sender": "System",
            "message": f"Client {client_id} left the chat session."
        })
