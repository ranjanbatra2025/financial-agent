# agent.py
import re
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from schema import AgentState
from stock_agent import stock_tool
from crypto_agent import crypto_tool
from forex_agent import forex_tool

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def classify_query(state: AgentState) -> AgentState:
    """Classify query into stocks, crypto, or forex using LLM"""
    if not llm:
        # Fallback to regex classification
        q_lower = state["messages"][-1].content.lower()
        if any(word in q_lower for word in ["stock", "share", "ticker"]) or re.search(r'\b[A-Z]{1,4}\b', state["messages"][-1].content):
            category = "stocks"
        elif any(word in q_lower for word in ["crypto", "bitcoin", "eth", "coin"]):
            category = "crypto"
        elif any(word in q_lower for word in ["forex", "currency", "usd", "eur", "convert"]):
            category = "forex"
        else:
            category = "stocks"  # Default
    else:
        prompt = ChatPromptTemplate.from_template(
            "Classify the query into one of: stocks, crypto, forex. Respond with only the category.\nQuery: {query}"
        )
        chain = prompt | llm | StrOutputParser()
        category = chain.invoke({"query": state["messages"][-1].content}).strip().lower()
        if category not in ["stocks", "crypto", "forex"]:
            category = "stocks"
    return {"messages": state["messages"], "category": category}

def route_to_tool(state: AgentState) -> AgentState:
    """Route to the appropriate tool based on category"""
    query = state["messages"][-1].content
    category = state["category"]
    if category == "stocks":
        result = stock_tool(query)
    elif category == "crypto":
        result = crypto_tool(query)
    elif category == "forex":
        result = forex_tool(query)
    else:
        result = "Could not classify query. Try specifying stocks, crypto, or forex."
    return {"messages": state["messages"] + [AIMessage(content=result)], "category": category, "result": result}

# Build Graph
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("classify", classify_query)
workflow.add_node("route_tool", route_to_tool)
workflow.set_entry_point("classify")
workflow.add_edge("classify", "route_tool")
workflow.add_edge("route_tool", END)

app_graph = workflow.compile()

def process_query(query: str) -> str:
    """Process query using LangGraph"""
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "category": None,
        "result": None
    }
    final_state = app_graph.invoke(initial_state)
    return final_state["result"]