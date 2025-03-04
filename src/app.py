# app.py - Updated Flask Backend

import os
import json
import time
import uuid
import logging
import threading
import networkx as nx
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, jsonify, session, Response, send_from_directory
from flask_session import Session
from pyvis.network import Network
import numpy as np
from werkzeug.local import LocalProxy
from functools import wraps
from contextlib import contextmanager

# Import agent module with improved error handling
try:
    from agent import agent as agent_module
    from agent.schemas import json_schemas
    OFFLINE_MODE = False
except ImportError:
    print("Warning: Agent module could not be imported. Running in offline mode.")
    OFFLINE_MODE = True

# Set up logging with Queue Handler for UI integration
class LogQueueHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.last_timestamp = 0
        
    def emit(self, record):
        log_entry = self.format(record)
        timestamp = time.time()
        self.logs.append((timestamp, log_entry))
        self.last_timestamp = timestamp

log_handler = LogQueueHandler()
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('agent_app')
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

# Configure working directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
CACHE_DIR = os.path.join(BASE_DIR, 'chat_cache')
GRAPH_DIR = os.path.join(CACHE_DIR, 'graphs')
SESSION_DIR = os.path.join(BASE_DIR, 'flask_session')

for directory in [STATIC_DIR, CACHE_DIR, GRAPH_DIR, SESSION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Flask app initialization
app = Flask(__name__, static_folder=STATIC_DIR)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = SESSION_DIR
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.secret_key = os.environ.get('SECRET_KEY', uuid.uuid4().hex)
Session(app)

# Thread-local storage for session replication in async contexts
thread_local = threading.local()

def generate_id(prefix=''):
    """Generate a unique ID with optional prefix"""
    return f"{prefix}{uuid.uuid4()}"

def load_default_settings(filename='defaults_session.json'):
    """Load default settings from JSON file"""
    try:
        with open(os.path.join(BASE_DIR, filename), 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading default settings: {e}")
        # Return basic default settings
        return {
            'session_id': None,
            'session_history': [],
            'agents_df': [
                {
                    'agent_id': 'agent_1',
                    'agent_name': 'Agent A',
                    'muted': False,
                    'description': 'An intelligent agent',
                    'environment': 'office',
                    'graph_file_path': None,
                    'persistance_count': 0,
                    'persistance_score': None,
                    'patience': 8,
                    'persistance': 3,
                    'last_response': '',
                    'last_narration': '',
                    'goal': 'Complete task A',
                    'current_aim': None,
                    'suggestion': '',
                    'current_node_location': 'start',
                    'personal_history': [],
                    'is_agent_generation_variables': True,
                    'generation_variables': {
                        'seed': 42,
                        'temperature': 0.7,
                        'max_tokens': 150,
                        'top_p': 0.9,
                        'use_gpu': True,
                        'llm': None
                    },
                    'impression_of_others': '',
                    'environment_changes': '',
                    'new_information': ''
                }
            ],
            'settings': {
                'seed': 42,
                'temperature': 0.7,
                'max_tokens': 150,
                'top_p': 0.9,
                'use_gpu': True
            },
            'len_last_history': 0,
            'user_name': 'User',
            'agent_mutes': [False]
        }

def initialize_session():
    """Initialize a new session with default settings"""
    logger.info("Initializing new session")
    
    defaults = load_default_settings()
    session_id = generate_id('session_')
    
    # Create agents with unique IDs and graph file paths
    agents = []
    for agent in defaults.get('agents_df', []):
        agent_id = generate_id('agent_')
        agent['agent_id'] = agent_id
        agent['graph_file_path'] = os.path.join(GRAPH_DIR, f"{agent_id}_graph.graphml")
        
        # Initialize empty graph
        graph = nx.DiGraph()
        graph.add_node('start')
        nx.write_graphml(graph, agent['graph_file_path'])
        
        agents.append(agent)
    
    # Create session state
    session_state = {
        'session_id': session_id,
        'user_name': defaults.get('user_name', 'User'),
        'agent_mutes': [False] * len(agents),
        'agents_df': agents,
        'session_history': [],
        'len_last_history': 0,
        'settings': defaults.get('settings', {}),
        'play': False,
        'is_user': True,
        'max_generations': 0,
        'current_generation': 0,
        'user_message': '',
        'number_of_agents': len(agents)
    }
    
    # Save to cache
    save_session_to_cache(session_state)
    
    # Update flask session
    session['state'] = session_state
    logger.info(f"Session initialized with ID: {session_id}")
    
    return session_state

def save_session_to_cache(session_state):
    """Save session state to cache file"""
    session_id = session_state['session_id']
    filepath = os.path.join(CACHE_DIR, f"{session_id}_state.json")
    
    # Convert DataFrame to list for JSON serialization if needed
    if isinstance(session_state.get('agents_df'), pd.DataFrame):
        session_state['agents_df'] = session_state['agents_df'].to_dict('records')
    
    with open(filepath, 'w') as f:
        json.dump(session_state, f, default=str)
    
    logger.info(f"Session state saved: {session_id}")

def load_session_from_cache(session_id):
    """Load session state from cache file"""
    filepath = os.path.join(CACHE_DIR, f"{session_id}_state.json")
    
    if not os.path.exists(filepath):
        logger.warning(f"Session file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Convert agents list to DataFrame if needed
        if isinstance(state.get('agents_df'), list):
            state['agents_df'] = pd.DataFrame(state['agents_df'])
        
        return state
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error loading session from cache: {e}")
        return None

def update_last_interaction(session_id):
    """Update the last interaction timestamp by touching the file"""
    filepath = os.path.join(CACHE_DIR, f"{session_id}_state.json")
    if os.path.exists(filepath):
        # Update the file's modification time
        os.utime(filepath, None)
        return True
    return False

def check_session_state():
    """Check if session state exists and initialize if needed"""
    if 'state' not in session:
        logger.warning("Session state not found, initializing new session")
        new_state = initialize_session()
        session['state'] = new_state
    return session['state']

def visualize_graph_pyvis(graph_file_path, output_path):
    """Generate a PyVis visualization of the agent's graph"""
    try:
        # Load the graph from GraphML file
        G = nx.read_graphml(graph_file_path)
        
        # Set up PyVis network
        net = Network(notebook=False, directed=True, cdn_resources='in_line')
        
        # Separate "Go" and "NoGo" edges
        go_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('label') == 'Go']
        nogo_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('label') == 'NoGo']
        
        # Extract weights
        go_weights = [float(G[u][v].get('weight', 1)) for u, v in go_edges]
        nogo_weights = [float(G[u][v].get('weight', 1)) for u, v in nogo_edges]
        
        # Calculate statistics for styling
        mean_weight = np.mean(go_weights) if go_weights else 1
        std_weight = np.std(go_weights) if go_weights else 0
        min_weight = min(nogo_weights) if nogo_weights else 0
        max_weight = max(nogo_weights) if nogo_weights else 1
        
        # Function for color scaling
        def weight_to_color(weight, min_val, max_val):
            if max_val == min_val:  # Avoid division by zero
                return 'lightgrey'
            normalized = (weight - min_val) / (max_val - min_val)
            grey_value = int(255 * (1 - normalized))
            return f'#{grey_value:02x}{grey_value:02x}{grey_value:02x}'
        
        # Add nodes to the network
        for node in G.nodes():
            # Style nodes differently based on incoming edges
            incoming_edges = G.in_edges(node, data=True)
            if any(d.get('label') == 'NoGo' for _, _, d in incoming_edges):
                weight = next(float(d.get('weight', 1)) for _, _, d in incoming_edges if d.get('label') == 'NoGo')
                color = weight_to_color(weight, min_weight, max_weight)
            else:
                color = '#4a90e2'  # Default blue color
            
            net.add_node(node, label=node, color=color)
        
        # Add edges to the network
        for u, v, data in G.edges(data=True):
            weight = float(data.get('weight', 1))
            label = data.get('label', '')
            
            if label == 'NoGo':
                net.add_edge(u, v, label=label, color='red', dashes=True, width=1.5)
            else:
                net.add_edge(u, v, label=label, color='green', width=2)
        
        # Customize network physics
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            },
            "minVelocity": 0.75
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "forceDirection": "none",
              "roundness": 0.2
            }
          },
          "nodes": {
            "font": {
              "size": 14
            },
            "borderWidth": 2,
            "shape": "dot",
            "size": 16
          }
        }
        """)
        
        # Save the network visualization
        net.save_graph(output_path)
        return True
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        return False

# API Routes
@app.route('/')
def index():
    """Render the main application page"""
    # Initialize session if needed
    check_session_state()
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory(STATIC_DIR, path)

@app.route('/get_session_id')
def get_session_id():
    """Get the current session ID"""
    check_session_state()
    return jsonify({"session_id": session['state']['session_id']})

@app.route('/check_session')
def check_session():
    """Check session status"""
    if 'state' in session:
        return jsonify({
            "session_id": session['state']['session_id'],
            "session_contents": session['state']
        })
    return jsonify({"session_id": None, "session_contents": None})

@app.route('/logs')
def get_logs():
    """Get logs since a specific timestamp"""
    since = float(request.args.get('since', 0))
    new_logs = []
    
    for timestamp, log in log_handler.logs:
        if timestamp > since:
            new_logs.append(log)
    
    current_timestamp = time.time()
    return jsonify({
        "logs": new_logs,
        "timestamp": current_timestamp
    })

@app.route('/submit', methods=['POST'])
def submit():
    """Submit a message and prepare for generation"""
    logger.info("Submit route called")
    check_session_state()
    
    # Get form data
    user_message = request.form.get('user_message', '')
    is_user = request.form.get('is_user', 'false').lower() == 'true'
    
    session_state = session['state']
    
    # If play is true, set it to false and return
    if session_state.get('play', False):
        session_state['play'] = False
        session['state'] = session_state
        return jsonify({
            "success": True,
            "play": False,
            "history": session_state['session_history']
        })
    
    # If user message, add to history
    if is_user and user_message:
        session_state['session_history'].append(("user", user_message))
        session_state['is_user'] = True
        session_state['user_message'] = user_message
    else:
        session_state['is_user'] = False
    
    # Set max generations and initialize generation process
    max_generations = len(session_state['agents_df']) if is_user else 100
    session_state['current_generation'] = 0
    session_state['max_generations'] = max_generations
    session_state['play'] = not is_user
    
    # Update session
    session['state'] = session_state
    
    # Update last interaction time
    update_last_interaction(session_state['session_id'])
    
    return jsonify({
        "success": True,
        "max_generations": max_generations,
        "history": session_state['session_history'],
        "play": session_state['play']
    })

@app.route('/generate')
def generate():
    """Generate a response from an agent"""
    if 'state' not in session:
        return jsonify({"error": "No active session"}), 400
    
    session_state = session['state']
    
    # Check if generation is stopped
    if not