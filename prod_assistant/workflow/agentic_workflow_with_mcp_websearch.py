# agentic_workflow_with_mcp_websearch.py
import sys
import asyncio
import platform
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

MCP_TOOL_TIMEOUT = 60  # seconds – prevent requests from hanging forever

from prompt_library.prompts import PROMPT_REGISTRY, PromptType

from utils.model_loader import ModelLoader
from langchain_mcp_adapters.client import MultiServerMCPClient
class AgenticRAG:
    """Agentic RAG pipeline using LangGraph + MCP (Retriever + WebSearch)."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # ---------- Initialization ----------
    def __init__(self):
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()

        # Use the standard sys.executable instead of the hacky _find_venv_python
        _venv_python = sys.executable
        _server_path = str(
            Path(__file__).resolve().parent.parent / "mcp_servers" / "product_search_server.py"
        )
        self.mcp_client = MultiServerMCPClient(
            {
                "hybrid_search": {
                    "command": _venv_python,
                    "args": [_server_path],
                    "transport": "stdio"
                }
            }
        )

        self.mcp_tools = []
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    async def _load_mcp_tools(self):
        """Load MCP tools. Must be called from an async context."""
        try:
            self.mcp_tools = await self.mcp_client.get_tools()
            print("MCP tools loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load MCP tools — {e}")
            self.mcp_tools = []

    @classmethod
    async def create(cls):
        """Async factory method: creates AgenticRAG and loads MCP tools."""
        instance = cls()
        await instance._load_mcp_tools()
        return instance

    def _get_latest_question(self, state: AgentState) -> str:
        """Extracts the latest question/query for the current turn, ignoring older history."""
        messages = state["messages"]
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            # Find the most recent AI tool call and use its query
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return msg.tool_calls[0]["args"].get("query", messages[i - 1].content)
        # Fallback to the last message if no tool call is found
        return messages[-1].content
    
    def _find_last_tool_call(self, messages):
        """Safely scans backward to find the ID of the last tool call."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                return msg.tool_calls[0]["id"]
        return "default_id"
    
    # ---------- Nodes ----------
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        
        sys_msg = SystemMessage(
            content="You are a helpful product assistant. When utilizing tools, you must output strictly valid JSON for your arguments without any extraneous characters."
        )
        
        # 1. The Bulletproof Filter: Keep ONLY Human questions and final AI text answers.
        clean_history = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                clean_history.append(msg)
            elif isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                # This explicitly ignores the Rewriter's internal steps and tool calls
                clean_history.append(msg)
                
        # 2. Sliding Window is now 100% safe because there are no orphaned tool sequences
        recent_history = clean_history[-6:]
        messages_to_pass = [sys_msg] + recent_history
        
        llm_with_tools = self.llm.bind_tools(self.mcp_tools)
        
        try:
            response = llm_with_tools.invoke(messages_to_pass)
            return {"messages": [response]}
        except Exception as e:
            print(f"API Tool Call Error: {e}")
            fallback_msg = AIMessage(
                content="I experienced a temporary formatting issue while trying to search for that. Could you please ask your question again?"
            )
            return {"messages": [fallback_msg]}

    async def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER (MCP) ---")
        query = self._get_latest_question(state)
        print(f"Retriever query: {query}")
        
        # Use the robust helper
        tool_id = self._find_last_tool_call(state["messages"])

        tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
        if not tool:
            return {"messages": [ToolMessage(content="Retriever tool not found.", tool_call_id=tool_id)]}

        try:
            result = await asyncio.wait_for(tool.ainvoke({"query": query}), timeout=MCP_TOOL_TIMEOUT)
            context = result or "No relevant product data found."
        except Exception as e:
            context = f"Error invoking retriever: {e}"

        return {"messages": [ToolMessage(content=context, tool_call_id=tool_id)]}

    async def _web_search(self, state: AgentState):
        print("--- WEB SEARCH (MCP) ---")
        query = state["messages"][-1].content
        query = query.replace("Rewritten Query: ","").strip()
        print(f"Searching for: {query}")
        
        # Use the robust helper
        tool_id = self._find_last_tool_call(state["messages"])
        
        tool = next((t for t in self.mcp_tools if t.name == "web_search"), None)
        if not tool:
            return {"messages": [ToolMessage(content="Web search tool not available.", tool_call_id=tool_id)]}
            
        try:
            result = await asyncio.wait_for(tool.ainvoke({"query": query}), timeout=MCP_TOOL_TIMEOUT)
            context = result if result else "No data from web"
        except Exception as e:
            context = f"Error during web search: {e}"
            
        print(f"Web result (first 300 chars): {context[:300]}")
        return {"messages": [ToolMessage(content=context, tool_call_id=tool_id)]}


    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = self._get_latest_question(state)
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a strict grader evaluating document relevance.
            Question: {question}
            Documents: {docs}
            
            If the documents contain ANY information related to the question, output exactly 'yes'.
            If the documents do not contain relevant information, output exactly 'no'.
            Do not provide any explanation.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs}).strip().lower()
        
        return "generator" if score == "yes" else "rewriter"

    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        
        # Scan backwards to find the exact text the human typed, not the tool query
        user_question = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "")
        
        docs = state["messages"][-1].content
        print(f"Using context (first 200 chars): {docs[:200]}")

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            # We pass the real user_question here
            response = chain.invoke({"context": docs, "question": user_question}) or "No response generated."
        except Exception as e:
            response = f"Error generating response: {e}"

        return {"messages": [AIMessage(content=response)]}

    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = self._get_latest_question(state)

        prompt = ChatPromptTemplate.from_template(
            "You are an SEO expert optimizing queries for an e-commerce search engine. "
            "Rewrite the user's query to find the most accurate pricing and product information. "
            "CRITICAL RULES:\n"
            "1. NEVER change the actual product name, brand, or model number.\n"
            "2. MUST PRESERVE all user-specified constraints like location (e.g., 'in India', 'in UK'), currency, or storage sizes. DO NOT append default currencies like USD unless the user explicitly asks for them.\n"
            "3. DO NOT add bracketed placeholders like [country] or [currency].\n"
            "4. Keep the query strictly in English.\n\n"
            "User Query: {question}\nRewritten Query:"
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            new_q = chain.invoke({"question": question}).strip()
        except Exception as e:
            new_q = f"Error rewriting query: {e}"

        return {"messages": [AIMessage(content=new_q)]}

    # ---------- Build Workflow ----------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("WebSearch", self._web_search)

        # Workflow edges
        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "WebSearch")
        workflow.add_edge("WebSearch", "Generator")

        return workflow

    # ---------- Public Run ----------
    async def run(self, query: str, thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content

# ---------- Standalone Test ----------
if __name__ == "__main__":
    import asyncio

    async def main():
        rag_agent = await AgenticRAG.create()
        answer = await rag_agent.run("What is the price of iPhone 16?")
        print("\nFinal Answer:\n", answer)

    asyncio.run(main())
