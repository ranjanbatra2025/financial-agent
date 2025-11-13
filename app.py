# app.py
from flask import Flask, request, jsonify, render_template
import os

from agent import process_query

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

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