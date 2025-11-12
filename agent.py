# agent.py
from flask import Flask, request, jsonify, render_template_string
import re
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from stock_agent import stock_tool
from crypto_agent import crypto_tool
from forex_agent import forex_tool

app = Flask(__name__)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# LangGraph State
class AgentState(TypedDict):
    messages: list
    category: str
    result: str

# Nodes
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

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Investment Assistant — Auto-Detect Mode</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0a0a0f;
      --bg-secondary: #1a1a24;
      --bg-card: #212130;
      --text-primary: #ffffff;
      --text-secondary: #b0b0b8;
      --accent-primary: #00d4ff;
      --accent-secondary: #6366f1;
      --border-color: #333344;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
      --border-radius: 12px;
      --transition: all 0.2s ease-in-out;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
      color: var(--text-primary);
      padding: 2rem 1rem;
      min-height: 100vh;
      line-height: 1.6;
    }

    .container {
      max-width: 800px;
      margin: 0 auto;
    }

    .header {
      text-align: center;
      margin-bottom: 2rem;
    }

    .header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.5rem;
    }

    .header p {
      color: var(--text-secondary);
      font-size: 1.1rem;
      font-weight: 400;
    }

    .card {
      background: var(--bg-card);
      border-radius: var(--border-radius);
      padding: 2rem;
      box-shadow: var(--shadow-lg);
      border: 1px solid var(--border-color);
      transition: var(--transition);
    }

    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    .form-group {
      margin-bottom: 1.5rem;
    }

    label {
      display: block;
      margin-bottom: 0.5rem;
      font-weight: 500;
      color: var(--text-primary);
      font-size: 0.95rem;
    }

    input[type="text"] {
      width: 100%;
      padding: 0.875rem 1rem;
      font-size: 1rem;
      border: 1px solid var(--border-color);
      border-radius: var(--border-radius);
      background: var(--bg-secondary);
      color: var(--text-primary);
      transition: var(--transition);
      font-family: inherit;
    }

    input[type="text"]:focus {
      outline: none;
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
    }

    .examples {
      margin-bottom: 1.5rem;
      padding: 1rem;
      background: rgba(0, 212, 255, 0.05);
      border-radius: var(--border-radius);
      border-left: 3px solid var(--accent-primary);
    }

    .examples code {
      background: var(--bg-secondary);
      padding: 0.25rem 0.5rem;
      border-radius: 6px;
      font-family: 'Monaco', monospace;
      color: var(--accent-primary);
    }

    .button-group {
      display: flex;
      gap: 0.75rem;
      margin-top: 1rem;
    }

    button {
      flex: 1;
      padding: 0.875rem 1.5rem;
      font-size: 1rem;
      font-weight: 500;
      border: none;
      border-radius: var(--border-radius);
      cursor: pointer;
      transition: var(--transition);
      font-family: inherit;
    }

    #ask {
      background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
      color: white;
    }

    #ask:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
    }

    #clear {
      background: var(--bg-secondary);
      color: var(--text-secondary);
      border: 1px solid var(--border-color);
    }

    #clear:hover {
      background: var(--border-color);
      color: var(--text-primary);
    }

    h3 {
      margin: 1.5rem 0 1rem;
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--text-primary);
    }

    #out {
      white-space: pre-wrap;
      background: var(--bg-secondary);
      padding: 1.5rem;
      border-radius: var(--border-radius);
      border: 1px solid var(--border-color);
      font-family: 'Inter', monospace;
      font-size: 0.95rem;
      line-height: 1.6;
      color: var(--text-primary);
      min-height: 120px;
      overflow-y: auto;
    }

    #out::before {
      content: '⟨ ';
      color: var(--accent-primary);
    }

    #out::after {
      content: ' ⟩';
      color: var(--accent-primary);
    }

    @media (max-width: 768px) {
      body {
        padding: 1rem 0.5rem;
      }

      .header h1 {
        font-size: 2rem;
      }

      .card {
        padding: 1.5rem;
      }

      .button-group {
        flex-direction: column;
      }
    }

    /* Loading animation */
    .loading {
      display: inline-block;
      width: 20px;
      height: 20px;
      border: 3px solid var(--bg-secondary);
      border-radius: 50%;
      border-top-color: var(--accent-primary);
      animation: spin 1s ease-in-out infinite;
      margin-right: 0.5rem;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Investment Assistant</h1>
      <p>Auto-Detect Intelligence for Stocks | Crypto | Forex</p>
    </div>
    <div class="card">
      <p style="color: var(--text-secondary); margin-bottom: 1.5rem; text-align: center;">
        LangGraph-powered auto-detection delivers precise insights. Enhanced by wisdom and real-time data from top APIs.
      </p>
      
      <div class="examples">
        <strong>Examples:</strong> <code>AAPL price</code>, <code>Apple</code>, <code>price of ETH</code>, <code>Bitcoin</code>, <code>Convert 100 USD to INR</code>
      </div>
      
      <div class="form-group">
        <input id="q" type="text" placeholder="Ask about stocks, crypto, or forex... (auto-detected)" />
      </div>
      
      <div class="button-group">
        <button id="ask">Ask <span id="loading" class="loading" style="display:none;"></span></button>
        <button id="clear">Clear</button>
      </div>
      
      <h3>Response</h3>
      <pre id="out">(Awaiting your query...)</pre>
    </div>
  </div>

<script>
const qInput = document.getElementById('q');
const out = document.getElementById('out');
const ask = document.getElementById('ask');
const clearBtn = document.getElementById('clear');
const loading = document.getElementById('loading');

async function send(query) {
  out.textContent = '⟨ Thinking... ⟩';
  loading.style.display = 'inline-block';
  ask.disabled = true;
  try {
    const resp = await fetch('/query', {
      method: 'POST', 
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({q: query})
    });
    const data = await resp.json();
    out.textContent = data.result;
  } catch (error) {
    out.textContent = '⟨ An error occurred. Please try again. ⟩';
  } finally {
    loading.style.display = 'none';
    ask.disabled = false;
  }
}

ask.addEventListener('click', () => {
  const q = qInput.value.trim();
  if (q) send(q);
});

qInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const q = qInput.value.trim();
    if (q) send(q);
  }
});

clearBtn.addEventListener('click', () => {
  qInput.value = '';
  out.textContent = '(Awaiting your query...)';
});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/query', methods=['POST'])
def query_route():
    data = request.get_json() or {}
    q = (data.get('q') or '').strip()
    if not q:
        return jsonify({'result': 'Please send a query.'})
    try:
        res = process_query(q)
        return jsonify({'result': res})
    except Exception as e:
        return jsonify({'result': f'Agent error: {str(e)}'})

if __name__ == '__main__':
    print('Starting Investment Assistant — LangGraph Auto-Detect Mode — open http://127.0.0.1:5000')
    print('Set POLYGON_API_KEY and OPENAI_API_KEY env vars for optimal performance.')
    app.run(debug=True)