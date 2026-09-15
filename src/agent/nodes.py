"""
Node implementations for the Legal Knowledge Assistant graph.

Unlike the earlier version of this graph, retrieval is no longer called
directly by a node -- it's exposed to the LLM as real tools
(document_search, document_metadata_lookup), and the model decides
whether and when to call them. This node module just wires that decision
loop together and enforces the two hard rules from the assignment:
  - bounded tool-call budget (agent.max_tool_calls in settings.yaml) so
    the loop can't run indefinitely
  - no answer is trusted as "grounded" unless a document_search tool call
    actually returned evidence during this run -- checked in `finalize`,
    not just asked of the model via prompt
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from src.config import get_settings, get_path
from src.llm.model import get_llm
from src.tools.retrieval_tool import make_document_search_tool
from src.tools.metadata_tool import make_metadata_lookup_tool
from .state import AgentState

from functools import lru_cache

logger = logging.getLogger(__name__)
_PROMPTS_DIR = get_path("prompts_dir")


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    return (_PROMPTS_DIR / "agent_system_prompt.txt").read_text()


def get_tools_for_scope(scope):
    """Tools are built per-scope so the agent can never retrieve content
    outside what the current user/session is authorized to see -- the
    scope is baked into each tool's closure at construction time."""
    return [make_document_search_tool(scope), make_metadata_lookup_tool(scope)]


def agent_node(state: AgentState) -> AgentState:
    """Calls the LLM with tools bound. The model decides whether it needs
    to call document_search / document_metadata_lookup, or has enough to
    answer already."""
    llm = get_llm()
    tools = get_tools_for_scope(state["scope"])
    llm_with_tools = llm.bind_tools(tools)

    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=_system_prompt())] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Routes to the tool node if the model asked for a tool call and the
    budget isn't exhausted, otherwise to finalize."""
    settings = get_settings()
    max_calls = settings["agent"]["max_tool_calls"]

    last = state["messages"][-1]
    calls_made = state.get("tool_calls_made", 0)

    wants_tools = isinstance(last, AIMessage) and bool(getattr(last, "tool_calls", None))
    if wants_tools and calls_made < max_calls:
        return "tools"
    return "finalize"


def count_tool_call(state: AgentState) -> AgentState:
    """Increments the tool-call budget counter after the tool node runs."""
    return {"tool_calls_made": state.get("tool_calls_made", 0) + 1}


def finalize(state: AgentState) -> AgentState:
    """Step 5/6 of the assignment's flow, done structurally rather than
    just by prompt: collect source references from any document_search
    results actually returned this run, and only treat the model's answer
    as grounded if such evidence exists. Otherwise return the fixed
    fallback message regardless of what the model said."""
    settings = get_settings()
    messages = state["messages"]

    sources: list[dict] = []
    saw_evidence = False
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "document_search":
            try:
                payload = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue
            for r in payload.get("results", []):
                saw_evidence = True
                sources.append(
                    {
                        "file_name": r["file_name"],
                        "page_number": r["page_number"],
                        "doc_id": r["doc_id"],
                    }
                )

    if not saw_evidence:
        logger.info("finalize: no document_search evidence retrieved, returning fallback")
        return {
            "answer": settings["agent"]["fallback_message"],
            "sources": [],
            "is_fallback": True,
        }

    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    answer = last_ai.content if last_ai else settings["agent"]["fallback_message"]

    return {"answer": answer, "sources": sources, "is_fallback": False}
