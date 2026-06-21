"""
SleepSense AI — ASI:One Connector (Fetch AI uAgents)

Registers on Agentverse in Mailbox mode so ASI:One can discover and
route natural-language sleep queries to SleepSense AI.

Run:
    AGENTVERSE_API_KEY=<key> \
    AGENT_SEED=<long-random-string> \
    NODE_SERVICE_URL=http://localhost:8000 \
    bin/bin/python agents/asi_one_connector.py

The agent will print its address (agent1q…) on startup.
Register it on https://agentverse.ai and add the README below
so ASI:One can discover it via semantic search.

Agentverse README tags: sleep health wearables eeg biomarkers
"""

import asyncio
import os
import sys

asyncio.set_event_loop(asyncio.new_event_loop())  # Python 3.14 requires explicit loop before uagents
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Configuration ────────────────────────────────────────────────────────────

AGENT_SEED    = os.environ.get("AGENT_SEED", "sleepsense_asi_connector_seed_2026_v1")
AGENT_PORT    = int(os.environ.get("AGENT_PORT", 8002))
SERVICE_URL   = os.environ.get("NODE_SERVICE_URL", "http://localhost:8000")
AGENTVERSE_KEY = os.environ.get("AGENTVERSE_API_KEY", "")

# ── Agent bootstrap ──────────────────────────────────────────────────────────

agent = Agent(
    name="sleepsense-ai",
    seed=AGENT_SEED,
    port=AGENT_PORT,
    mailbox=bool(AGENTVERSE_KEY),
    agentverse={
        "api_key": AGENTVERSE_KEY,
    } if AGENTVERSE_KEY else {},
    endpoint=[f"http://localhost:{AGENT_PORT}/submit"],
)

chat_protocol = Protocol(spec=chat_protocol_spec)


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _query_backend(user_address: str, text: str) -> str:
    """Forward the user's query to the FastAPI backend and return the reply."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{SERVICE_URL}/api/agent/query",
            json={"user_id": user_address, "query": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("summary", "Sorry, I could not retrieve your sleep data right now.")


def _make_reply(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )


# ── Chat Protocol handler ────────────────────────────────────────────────────

@chat_protocol.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass  # acks from ASI:One — no action needed


@chat_protocol.on_message(ChatMessage)
async def on_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    # Always ack immediately
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(timezone.utc),
        acknowledged_msg_id=msg.msg_id,
    ))

    # Extract text from the message
    text = " ".join(
        c.text for c in msg.content
        if isinstance(c, TextContent) and c.text.strip()
    ).strip()

    if not text:
        await ctx.send(sender, _make_reply(
            "Hi! I'm SleepSense AI. Ask me anything about your sleep — "
            "e.g. 'How did I sleep last night?' or 'What's my sleep debt?'"
        ))
        return

    ctx.logger.info(f"Query from {sender[:20]}…: {text[:80]}")

    try:
        reply = await _query_backend(sender, text)
    except httpx.HTTPStatusError as e:
        ctx.logger.error(f"Backend error {e.response.status_code}: {e.response.text}")
        reply = "SleepSense backend is unavailable right now. Please try again in a moment."
    except Exception as e:
        ctx.logger.error(f"Unexpected error: {e}")
        reply = "Something went wrong processing your request. Please try again."

    await ctx.send(sender, _make_reply(reply))


agent.include(chat_protocol, publish_manifest=False)


# ── Startup log ──────────────────────────────────────────────────────────────

@agent.on_event("startup")
async def on_startup(ctx: Context):
    ctx.logger.info(f"SleepSense AI agent started")
    ctx.logger.info(f"Address : {agent.address}")
    ctx.logger.info(f"Backend : {SERVICE_URL}")
    ctx.logger.info(f"Mailbox : {'enabled' if AGENTVERSE_KEY else 'disabled (set AGENTVERSE_API_KEY)'}")


if __name__ == "__main__":
    print(f"\nSleepSense AI — ASI:One Connector")
    print(f"Agent address : {agent.address}")
    print(f"Backend URL   : {SERVICE_URL}")
    print(f"Mailbox mode  : {'ON' if AGENTVERSE_KEY else 'OFF (set AGENTVERSE_API_KEY to enable)'}")
    print(f"Port          : {AGENT_PORT}\n")
    agent.run()
