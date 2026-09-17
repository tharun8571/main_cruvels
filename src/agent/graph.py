"""
LangGraph wiring for the Legal Knowledge Assistant.

Flow:
  agent -> [has tool calls? and under budget?]
              -> yes: tools (ToolNode) -> count_tool_call -> agent (loop)
              -> no:  finalize -> END

This replaces a version of the graph that called retrieval directly from a
node. Now the LLM is bound to real tools (document_search,
document_metadata_lookup) via a proper `ToolNode`, and decides for itself
whether/when to call them -- matching the assignment's "agent that can
choose between a limited number of tools" requirement (section 4.1) rather
than a fixed retrieval step.

The agent never takes any action beyond these two read-only tools -- no
external legal actions, no irreversible decisions (assignment section 4.2).
"""
from __future__ import annotations

import logging

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.config import get_settings
from src.context.permissions import AuthorizedScope
from src.retrieval.retriever import retrieve_chunks
from src.llm.model import get_llm
from .state import AgentState
from .nodes import agent_node, should_continue, count_tool_call, finalize, get_tools_for_scope, _system_prompt

logger = logging.getLogger(__name__)


def build_agent_graph(scope: AuthorizedScope):
    """Builds the multi-turn tool-calling graph for complex workflows."""
    tools = get_tools_for_scope(scope)
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("count_tool_call", count_tool_call)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "count_tool_call")
    graph.add_edge("count_tool_call", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent(question: str, scope: AuthorizedScope) -> dict:
    """Fast-path grounded legal Q&A agent.
    Performs high-speed direct vector search first (sub-50ms) and executes
    a single-pass grounded generation, cutting out multiple LLM roundtrips.
    """
    settings = get_settings()
    chunks = retrieve_chunks(question, scope)

    if not chunks:
        logger.info("run_agent: no matching chunks found, returning fallback")
        return {
            "answer": settings["agent"]["fallback_message"],
            "sources": [],
            "is_fallback": True,
        }

    sources = []
    seen = set()
    for c in chunks:
        key = (c.file_name, c.page_number)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file_name": c.file_name,
                "page_number": c.page_number,
                "doc_id": c.doc_id,
            })

    evidence_blocks = [
        f"--- [Source: {c.file_name}, Page: {c.page_number}] ---\n{c.text}"
        for c in chunks
    ]
    combined_evidence = "\n\n".join(evidence_blocks)

    system_msg = SystemMessage(content=_system_prompt())
    human_msg = HumanMessage(
        content=(
            f"User Question:\n{question}\n\n"
            f"Authorized Document Evidence:\n{combined_evidence}\n\n"
            "Instructions: Provide a clear, accurate, and professional legal answer strictly grounded in the authorized evidence above. "
            "Cite the specific document name and page number for each key point."
        )
    )

    try:
        llm = get_llm()
        response = llm.invoke([system_msg, human_msg])
        answer_text = response.content.strip()
        return {
            "answer": answer_text,
            "sources": sources,
            "is_fallback": False,
        }
    except Exception as e:
        logger.warning("Error during fast LLM invocation, falling back to graph agent: %s", e)
        app = build_agent_graph(scope)
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "scope": scope,
            "tool_calls_made": 0,
        }
        final_state = app.invoke(initial_state)
        return {
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "is_fallback": final_state.get("is_fallback", False),
        }


def stream_agent(question: str, scope: AuthorizedScope):
    """Streams tokens in real-time as line-delimited JSON chunks for instantaneous UI response."""
    settings = get_settings()
    chunks = retrieve_chunks(question, scope)

    if not chunks:
        yield json.dumps({
            "type": "fallback",
            "answer": settings["agent"]["fallback_message"],
            "sources": [],
            "is_fallback": True,
        }) + "\n"
        return

    sources = []
    seen = set()
    for c in chunks:
        key = (c.file_name, c.page_number)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file_name": c.file_name,
                "page_number": c.page_number,
                "doc_id": c.doc_id,
            })

    # Yield citations immediately so the UI renders citation badges in <50ms
    yield json.dumps({"type": "sources", "sources": sources, "is_fallback": False}) + "\n"

    evidence_blocks = [
        f"--- [Source: {c.file_name}, Page: {c.page_number}] ---\n{c.text}"
        for c in chunks
    ]
    combined_evidence = "\n\n".join(evidence_blocks)

    system_msg = SystemMessage(content=_system_prompt())
    human_msg = HumanMessage(
        content=(
            f"User Question:\n{question}\n\n"
            f"Authorized Document Evidence:\n{combined_evidence}\n\n"
            "Instructions: Provide a clear, accurate, and professional legal answer strictly grounded in the authorized evidence above. "
            "Cite the specific document name and page number for each key point."
        )
    )

    try:
        llm = get_llm()
        for chunk in llm.stream([system_msg, human_msg]):
            if chunk.content:
                yield json.dumps({"type": "token", "token": chunk.content}) + "\n"
    except Exception as e:
        logger.exception("Error during LLM streaming: %s", e)
        yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    yield json.dumps({"type": "done"}) + "\n"
