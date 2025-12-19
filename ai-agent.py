import os
import logging
from typing import TypedDict, List, Annotated
from operator import add
from pathlib import Path
from dotenv import load_dotenv

# Third party stuff
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Config - keeping it simple, no fancy class
MODEL_NAME = "llama-3.3-70b-versatile"  # Groq's beast
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = Path("./data/agent_memory_db")
MAX_SEARCH_RESULTS = 3
MAX_REVS = 2  # shorter name, cuz why not

# Logging setup - I like this better than prints for prod
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentMahashay")

load_dotenv()  # env vars, duh

# Init - with some error handling, but not overkill
llm = None
embeddings = None
vector_store = None
web_search_tool = None

try:
    logger.info("Firing up the AI brain (Groq)...")
    llm = ChatGroq(model=MODEL_NAME, temperature=0)

    logger.info("Grabbing local embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info("Hooking into Chroma DB...")
    vector_store = Chroma(
        collection_name="agent_knowledge",
        persist_directory=str(DB_PATH),
        embedding_function=embeddings
    )
    
    # Tools
    web_search_tool = TavilySearch(max_results=MAX_SEARCH_RESULTS)  # dropped topic, it's general anyway

except Exception as e:
    logger.error(f"Oof, init blew up: {e}")
    raise

# State - TypedDict is nice, but humans don't always annotate everything
class AgentState(TypedDict):
    task: str
    research_data: List[str]
    draft: str
    critique: str
    rev_num: int  # shorter, more casual
    history: Annotated[List[BaseMessage], add]

# Nodes - keeping 'em functional

def check_safety(state):
    """Quick safety net - no exploits on my watch."""
    query = state.get("task", "").lower()
    bad_words = ["exploit", "hack", "illegal", "bypass"]  # add more as needed
    
    if any(word in query for word in bad_words):
        logger.warning(f"Nope, blocking: {query}")
        return {"draft": "BLOCKED_BY_POLICY", "history": []}
    
    return {"history": [SystemMessage(content="All good, proceed.")]}

def research_task(state):
    """Dig into memory and web for deets."""
    logger.info(f"Researching: {state['task']}")
    print("🔍 Hunting knowledge...")  # sneaky print for user feedback
    
    # RAG first - reuse old wins
    existing = vector_store.similarity_search(state['task'], k=1)
    internal = f"From memory: {existing[0].page_content}" if existing else "Nada in the vault."
    
    # Web hunt
    try:
        web_results = web_search_tool.invoke({"query": state['task']})
        if isinstance(web_results, list):
            external = [f"Web hit: {res.get('content', str(res))}" for res in web_results]
        else:
            external = [f"Web hit: {str(web_results)}"]
    except Exception as e:
        logger.error(f"Web search flopped: {e}")
        external = ["Web down, sticking to internals."]

    return {"research_data": [internal] + external}

def draft_content(state):
    """Whip up a draft from the mess of research."""
    rev = state.get("rev_num", 0) + 1
    logger.info(f"Draft round {rev}...")
    
    prompt = f"""
    You're a sharp tech analyst. Boil this research down to a solid report.
    Research bits: {state['research_data']}
    Last notes (if any): {state.get('critique', 'Starting fresh')}
    Keep it pro but concise.
    """
    response = llm.invoke(prompt)
    return {"draft": response.content, "rev_num": rev}

def critique_draft(state):
    """Poke holes or greenlight the draft."""
    logger.info("Review time...")
    
    prompt = f"""
    Scan this draft:
    {state['draft']}
    
    If it's on point and covers the task, just say 'APPROVED'.
    Otherwise, bullet out fixes needed. Be picky but fair.
    """
    response = llm.invoke(prompt)
    return {"critique": response.content}

def archive_content(state):
    """Stash the winner in DB."""
    logger.info("Locking it down in memory...")
    doc = Document(page_content=state['draft'], metadata={"source": "agent_workflow"})
    vector_store.add_documents([doc])
    return {"history": [SystemMessage(content="Saved for posterity.")]}

# Graph builder - the magic flow
def build_graph():
    wf = StateGraph(AgentState)

    wf.add_node("safety", check_safety)
    wf.add_node("research", research_task)
    wf.add_node("write", draft_content)
    wf.add_node("review", critique_draft)
    wf.add_node("archive", archive_content)

    wf.add_edge(START, "safety")
    
    # Safety router
    def safety_route(state):
        return END if state.get("draft") == "BLOCKED_BY_POLICY" else "research"
    wf.add_conditional_edges("safety", safety_route)

    wf.add_edge("research", "write")
    wf.add_edge("write", "review")

    # Review loop - bail after max or approved
    def review_route(state):
        critique = state['critique'].upper()
        if "APPROVED" in critique or state['rev_num'] >= MAX_REVS:
            return "archive"
        return "write"  # loop back
    wf.add_conditional_edges("review", review_route)

    wf.add_edge("archive", END)
    
    return wf.compile(checkpointer=MemorySaver())

# Run it
if __name__ == "__main__":
    app = build_graph()
    print("\n🚀 Agent Mahashay ready to roll (Groq + local smarts)")
    
    while True:
        try:
            user_query = input("\nYour ask > ").strip()
            if user_query.lower() in ["exit", "quit", "bye"]:
                print("Catch ya later!")
                break
                
            config = {"configurable": {"thread_id": "session_1"}}
            
            print("--- Cranking the gears ---")
            for _ in app.stream({"task": user_query, "rev_num": 0}, config):
                pass  # logs do the talking
            
            # Grab the goods
            final_state = app.get_state(config)
            draft = final_state.values.get("draft")
            if draft and draft != "BLOCKED_BY_POLICY":
                print("\n" + "="*50)
                print("📄 Your Report:")
                print("="*50)
                print(draft)
                print("="*50 + "\n")
            else:
                print("\n🚫 Safety says no-go.")
                
        except KeyboardInterrupt:
            print("\nAborted - no harm done.")
            break
        except Exception as err:
            logger.error(f"Loop crash: {err}")
            print("Something broke, try again?")