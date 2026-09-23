"""Conversations and the chat surface, including its socket."""

from __future__ import annotations

import json

from ..schemas import ChatRequest, ConversationCreate, ConversationUpdate
from .services import AppServices
from fastapi import HTTPException
from fastapi import Query
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import Any


def register_conversation_routes(application: FastAPI, sv: AppServices) -> None:
    settings = sv.settings
    database = sv.database
    agent = sv.agent
    @application.get("/api/conversations")
    async def list_conversations(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {"conversations": database.list_conversations(limit)}

    @application.post("/api/conversations", status_code=201)
    async def create_conversation(payload: ConversationCreate) -> dict[str, Any]:
        default_models = {
            "auto": "auto",
            "ollama": settings.default_model,
            "litellm": settings.litellm_fast_model,
            "openrouter": (
                settings.default_model
                if "/" in str(settings.default_model or "") and not str(settings.default_model).lower().startswith("qwen")
                else settings.openrouter_fast_model
            ),
            "openai": settings.openai_model,
        }
        model = payload.model or default_models[payload.provider]
        conversation = database.create_conversation(payload.title, payload.provider, model)
        database.add_audit("conversation", "created", "Conversation created", actor="user", conversation_id=conversation["id"])
        return {"conversation": conversation}

    @application.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str) -> dict[str, Any]:
        conversation = database.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        return {"conversation": conversation, "messages": database.list_messages(conversation_id)}

    @application.patch("/api/conversations/{conversation_id}")
    async def update_conversation(conversation_id: str, payload: ConversationUpdate) -> dict[str, Any]:
        conversation = database.update_conversation_title(conversation_id, payload.title)
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        database.add_audit(
            "conversation",
            "renamed",
            "Conversation renamed",
            actor="user",
            conversation_id=conversation_id,
            details={"conversation_id": conversation_id, "title_characters": len(conversation["title"])},
        )
        return {"conversation": conversation}

    @application.get("/api/conversations/{conversation_id}/messages")
    async def conversation_messages(conversation_id: str, limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
        if database.get_conversation(conversation_id) is None:
            raise HTTPException(404, "Conversation not found")
        return {"messages": database.list_messages(conversation_id, limit)}

    @application.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str) -> dict[str, Any]:
        if not database.delete_conversation(conversation_id):
            raise HTTPException(404, "Conversation not found")
        database.add_audit("conversation", "deleted", "Conversation and its messages deleted", actor="user", details={"conversation_id": conversation_id})
        return {"deleted": True, "id": conversation_id}

    @application.post("/api/chat")
    async def chat(payload: ChatRequest) -> dict[str, Any]:
        try:
            return await agent.chat(
                payload.message, conversation_id=payload.conversation_id, provider=payload.provider, model=payload.model,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @application.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest) -> StreamingResponse:
        async def events():
            yield "event: status\ndata: " + json.dumps({"status": "thinking"}) + "\n\n"
            try:
                result = await agent.chat(
                    payload.message, conversation_id=payload.conversation_id, provider=payload.provider, model=payload.model,
                )
                yield "event: result\ndata: " + json.dumps(result, ensure_ascii=False, default=str) + "\n\n"
            except Exception as exc:
                yield "event: error\ndata: " + json.dumps({"error": str(exc)}) + "\n\n"
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @application.websocket("/ws/chat")
    async def chat_socket(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        client_host = (websocket.client.host if websocket.client else "").lower()
        if (origin and origin not in settings.cors_origins) or client_host not in {"127.0.0.1", "::1", "testclient"}:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                payload = ChatRequest.model_validate(await websocket.receive_json())
                await websocket.send_json({"type": "status", "status": "thinking"})
                result = await agent.chat(
                    payload.message, conversation_id=payload.conversation_id, provider=payload.provider, model=payload.model,
                )
                await websocket.send_json({"type": "result", **result})
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_json({"type": "error", "error": str(exc)})
