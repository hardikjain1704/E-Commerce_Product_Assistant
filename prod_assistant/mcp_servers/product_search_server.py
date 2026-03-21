# product_search_server.py
from mcp.server.fastmcp import FastMCP
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# Initialize MCP server (lightweight — no network calls)
mcp = FastMCP("hybrid_search")

# Lazy-loaded singletons — initialised on first tool call so the MCP
# server can register its tools and start the stdio transport immediately,
# even if AstraDB is temporarily unreachable.
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from retriever.retrieval import Retriever
        _retriever = Retriever().load_retriever()
    return _retriever


_tavily = None

def _get_tavily():
    global _tavily
    if _tavily is None:
        _tavily = TavilySearchResults(max_results=3)
    return _tavily


# ---------- Helpers ----------
def format_docs(docs) -> str:
    """Format retriever docs into readable context."""
    if not docs:
        return ""
    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)
    return "\n\n---\n\n".join(formatted_chunks)

# ---------- MCP Tools ----------
@mcp.tool()
async def get_product_info(query: str) -> str:
    """Retrieve product information for a given query from local retriever."""
    try:
        retriever = _get_retriever()
        docs = retriever.invoke(query)
        context = format_docs(docs)
        if not context.strip():
            return "No local results found."
        return context
    except Exception as e:
        return f"Error retrieving product info: {str(e)}"

@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web using Tavily if retriever has no results."""
    try:
        results = _get_tavily().invoke({"query": query})
        if not results:
            return "No data found on the web."
            
        formatted_results = []
        for res in results:
            formatted_results.append(f"Source: {res.get('url', 'N/A')}\nContent: {res.get('content', 'N/A')}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error during web search: {str(e)}"

# ---------- Run Server ----------
if __name__ == "__main__":
    mcp.run(transport="stdio")
    #mcp.run(transport="streamable-http")
