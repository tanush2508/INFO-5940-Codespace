"""
Multi-Agent Travel Planner (clean version, no background image)
"""

from __future__ import annotations
import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────
# Environment Setup
# ──────────────────────────────────────────────────────────────

load_dotenv()
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            pass


def redact_for_logs(value: Any) -> Any:
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────
# Agent Framework Imports
# ──────────────────────────────────────────────────────────────

from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """Internet search backed by Tavily."""
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────
# Agent Prompts
# ──────────────────────────────────────────────────────────────

REVIEWER_INSTRUCTIONS = """
You are a friendly, experienced travel reviewer. 
Your role is to read the travel plan, verify if it’s realistic, safe, and logistically sound — and make it feel inspiring.

Steps:
1. Validate all recommendations (destinations, hotels, attractions, timing).
2. Ensure the budget, season, and travel flow make sense.
3. Use your natural writing voice — helpful, creative, like a well-traveled friend.
4. If something doesn’t add up, fix it kindly and explain why.
5. Keep the final output structured by days or themes.

Goal: A polished, reliable, and exciting plan that the user would love to follow.
"""

PLANNER_INSTRUCTIONS = """
You are a creative travel planner with great taste.
Design personalized itineraries that balance fun, rest, food, and adventure.

Guidelines:
- Think like a human who loves to explore — not an algorithm.
- Respect the user's budget, duration, and interests.
- Keep a logical flow: minimize travel time, include meals and downtime.
- Use descriptive language (“Start your morning with a croissant at a quiet café…”).
- Output should be structured clearly by day or activity group.

Goal: A plan that feels handcrafted — smart, realistic, and full of life.
"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search],
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)


# ──────────────────────────────────────────────────────────────
# Async-safe Runner Helpers
# ──────────────────────────────────────────────────────────────

async def run_agent_async(agent: Agent, input_text: str):
    result = await Runner.run(agent, input_text)
    return getattr(result, "final_output", None) or getattr(result, "text", None) or str(result)


def run_agent(agent: Agent, text: str):
    """Runs agents safely even inside Streamlit's event loop."""
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(run_agent_async(agent, text))
    except RuntimeError:
        return asyncio.run(run_agent_async(agent, text))


# ──────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️", layout="centered")

st.title("🌍 Multi-Agent Travel Planner")
st.caption("_Planner → Reviewer: Smart, creative, and ready for adventure._")

# Sidebar controls
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    show_tools = st.toggle("Show live tool activity", value=True)
    if show_tools:
        with st.expander("🔧 Tool activity (live)", expanded=False):
            tool_panel = st.container()
    else:
        tool_panel = st.container()

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "meta" not in st.session_state:
    st.session_state.meta = []

# Render past chat
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            st.caption(st.session_state.meta[i].get("trace", ""))

# Chat input
user_input = st.chat_input("Where would you like to go and what’s your vibe?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        live_msg = st.empty()
        progress = st.progress(0)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:
                    t, et = ev.get("tool", "unknown"), ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        set_tool_logger(ui_tool_logger)

        try:
            live_msg.markdown("🧭 Planner Agent is sketching your adventure…")
            plan_text = run_agent(planner_agent, user_input)
            progress.progress(40)

            live_msg.markdown("🔎 Reviewer Agent is double-checking and refining your plan…")
            review_text = run_agent(reviewer_agent, plan_text)
            progress.progress(90)

            live_msg.markdown("✅ All set! Here’s your polished itinerary:")
            time.sleep(0.2)
            progress.progress(100)

            st.success("✨ **Final Travel Plan** (Reviewed and Ready)")
            st.markdown(review_text)
            with st.expander("See original draft from Planner Agent"):
                st.markdown(plan_text)

            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})

        except Exception as e:
            live_msg.markdown("❌ Oops, something went wrong.")
            err = f"⚠️ There was an error while preparing your trip:\n\n```\n{e}\n```"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            set_tool_logger(None)
