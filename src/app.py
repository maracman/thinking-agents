import threading
import socket
import time
import re
import queue
import logging
import subprocess
import os
import json
import uuid
import networkx as nx
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template, jsonify, session, Response, redirect, url_for, send_from_directory
from flask_session import Session
from werkzeug.local import LocalProxy
from functools import wraps
from pyvis.network import Network
import pandas as pd
from agent.schemas import json_schemas
import importlib

# Set up logging
logging.basicConfig(level=logging.DEBUG)
flask_logger = logging.getLogger('flask_app')

# Create necessary directories
os.makedirs('chat_cache', exist_ok=True)
os.makedirs('chat_cache/graphs', exist_ok=True)
os.makedirs('flask_session', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global variables
global_log_queue = queue.Queue()
offline = True  # Set to False for production using real LLM
VERSION = time.time()
current_device = None

# Create a QueueHandler for logging
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

def setup_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Queue Handler
        queue_handler = QueueHandler(global_log_queue)
        queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(queue_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    return logger

# Setup the Flask logger
flask_logger = setup_logger('flask_app')
logger = setup_logger('agent_logger')

def get_logs():
    logs = []
    while not global_log_queue.empty():
        logs.append(global_log_queue.get())
    return '\n'.join(logs)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24).hex())
Session(app)

# Cache directory path
cache_dir = os.path.join(os.getcwd(), 'chat_cache')
graph_directory = os.path.join(cache_dir, 'graphs')

# Model configuration
local_model_path = "/content/local_model"
model_filename = os.path.join(local_model_path, "Meta-Llama-3-8B.Q8_0.gguf")
model_url = "https://huggingface.co/TheBloke/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B.Q8_0.gguf?download=true"

def download_model():
    """Download the model if it doesn't exist"""
    os.makedirs(local_model_path, exist_ok=True)
    import requests
    
    response = requests.get(model_url)
    with open(model_filename, 'wb') as model_file:
        model_file.write(response.content)

    flask_logger.info(f"Model downloaded and saved to {model_filename}")

    # Create a barebones config file for the model
    config = {"model_type": "llama"}
    config_path = os.path.join(local_model_path, "config.json")
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file)
    flask_logger.info(f"Config file saved at {config_path}")

def reload_agent():
    """Reload the agent module dynamically"""
    try:
        if 'agent' in sys.modules:
            del sys.modules['agent']
        script_path = os.path.join('agent', 'agent.py')
        spec = importlib.util.spec_from_file_location("agent", script_path)
        agent = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = agent
        spec.loader.exec_module(agent)
        flask_logger.info("Agent module reloaded successfully")
        return agent
    except Exception as e:
        flask_logger.error(f"Failed to reload agent: {str(e)}")
        raise

def visualize_graph_pyvis(graph_file_path, session_id):
    """Generate a visualization of the agent's decision graph"""
    try:
        # Load the graph from the GraphML file
        G = nx.read_graphml(graph_file_path)

        net = Network(notebook=True, directed=True, cdn_resources='in_line')

        # Separate "Go" and "NoGo" edges
        go_edges = [(u, v) for u, v, d in G.edges(data=True) if d['label'] == 'Go']
        nogo_edges = [(u, v) for u, v, d in G.edges(data=True) if d['label'] == 'NoGo']

        # Extract weights for "Go" edges
        go_weights = [G[u][v]['weight'] for u, v in go_edges]

        # Calculate the mean and standard deviation for "Go" weights
        mean_weight = np.mean(go_weights) if go_weights else 0
        std_weight = np.std(go_weights) if go_weights else 0

        # Calculate the length for "NoGo" edges
        nogo_length = (mean_weight - (std_weight * 2)) * 100 if mean_weight and std_weight else 100

        # Extract weights for "NoGo" edges
        nogo_weights = [G[u][v]['weight'] for u, v in nogo_edges]

        # Calculate the range for "NoGo" weights
        min_weight = min(nogo_weights, default=0)
        max_weight = max(nogo_weights, default=1)

        # Function to calculate color based on weight
        def weight_to_color(weight, min_weight, max_weight):
            if max_weight == min_weight:  # Avoid division by zero
                return 'lightgrey'
            normalized = (weight - min_weight) / (max_weight - min_weight)
            grey_value = int(255 * (1 - normalized))
            return f'#{grey_value:02x}{grey_value:02x}{grey_value:02x}'

        # Add nodes and edges to the network
        for node in G.nodes():
            incoming_edges = G.in_edges(node, data=True)
            if any(d['label'] == 'NoGo' for u, v, d in incoming_edges):
                weight = next(d['weight'] for u, v, d in incoming_edges if d['label'] == 'NoGo')
                color = weight_to_color(weight, min_weight, max_weight)
            else:
                color = 'blue'
            net.add_node(node, label=node, color=color)

        for u, v, data in G.edges(data=True):
            weight = data['weight']
            if data['label'] == 'NoGo':
                net.add_edge(u, v, label=data['label'], weight=weight, length=nogo_length)
            else:
                net.add_edge(u, v, label=data['label'], weight=weight, length=weight * 100)

        # Customize physics to better reflect distances
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
          }
        }
        """)

        # Generate a unique filename for this graph visualization
        timestamp = int(time.time() * 1000)
        filename = f"graph_{session_id}_{timestamp}.html"

        # Save the graph visualization
        net.save_graph(os.path.join(app.static_folder, filename))

        return filename
    except Exception as e:
        flask_logger.error(f"Error visualizing graph: {str(e)}")
        return None

# Load default settings for initializing a new session
def load_default_settings(filename='defaults_session.json'):
    with open(filename, 'r') as file:
        return json.load(file)

# Generate unique session ID
def generate_session_id():
    return f"session_{uuid.uuid4()}"

# Generate unique agent ID
def generate_agent_id():
    return f"agent_{uuid.uuid4()}"

# Initialize agents DataFrame from defaults
def initialize_agents_df(default_agents):
    agents = []
    for agent in default_agents:
        agent['agent_id'] = generate_agent_id()
        agent['graph_file_path'] = os.path.join(graph_directory, f"{agent['agent_id']}_graph.graphml")
        # Initialize graph
        graph = nx.DiGraph()
        graph.add_node('start')
        nx.write_graphml(graph, agent['graph_file_path'])
        agents.append(agent)
    return pd.DataFrame(agents)

# Initialize single agent
def initialize_single_agent(default_agent):
    agent_id = generate_agent_id()
    default_agent['agent_id'] = agent_id
    default_agent['graph_file_path'] = os.path.join(graph_directory, f"{agent_id}_graph.graphml")
    # Initialize graph
    graph = nx.DiGraph()
    graph.add_node('start')
    nx.write_graphml(graph, default_agent['graph_file_path'])
    return default_agent

def check_session_state(passed_function="not specified"):
    flask_logger.debug(f"Session contents: {session}")
    flask_logger.info(f"check_session_state called by function: {passed_function}")
    
    if 'state' not in session:
        flask_logger.warning("Session state not found, initializing new session")
        new_state = initialize_session()
        session['state'] = new_state
        flask_logger.debug(f"New session state initialized: {new_state}")
    else:
        flask_logger.debug("Session state found")
    
    return session['state']

def initialize_session():
    flask_logger.debug("Entering initialize_session function")
    
    if 'state' in session and session['state'].get('session_id'):
        flask_logger.info(f"Valid session already exists with ID: {session['state']['session_id']}")
        return session['state']

    defaults = load_default_settings()
    flask_logger.debug(f"Loaded default settings: {defaults}")

    session_id = generate_session_id()
    agents_df = initialize_agents_df(defaults['agents_df'])

    session_state = {
        'session_id': session_id,
        'user_name': defaults['user_name'],
        'agent_mutes': [False] * len(agents_df),
        'agents_df': agents_df.to_dict('records'),
        'session_history': [],
        'len_last_history': defaults['len_last_history'],
        'settings': defaults['settings'],
        'play': False,
        'is_user': False,
        'max_generations': 0,
        'current_generation': 0,
        'user_message': '',
        'number_of_agents': len(agents_df)
    }

    flask_logger.info(f"New session initialized with ID: {session_id}")
    flask_logger.debug(f"Initial session state: {session_state}")
    save_current_session(session_state)
    session['state'] = session_state
    flask_logger.info(f"New session initialized, session state: {session['state']}")
    flask_logger.info(f"Checking persistence of session state")
    check_session_state()
    flask_logger.info("Exiting initialize_session function")
    return session_state

def save_current_session(session_state):
    session_id = session_state['session_id']
    filename = f"{session_id}_state.json"
    filepath = os.path.join(cache_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(session_state, f)
    flask_logger.info(f"Session state saved: {session_id}")

def load_state(session_id):
    filename = f"{session_id}_state.json"
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
            return state
    return None

# Adding a single agent to an existing session
def add_single_agent_to_session(default_agent):
    if 'state' not in session or not session['state']:
        flask_logger.error("No session found to add an agent to")
        return

    new_agent = initialize_single_agent(default_agent)
    agents_df = pd.DataFrame(session['state']['agents_df'])
    agents_df = pd.concat([agents_df, pd.DataFrame([new_agent])], ignore_index=True)
    session['state']['agents_df'] = agents_df.to_dict('records')
    session['state']['agent_mutes'].append(False)
    session['state']['number_of_agents'] += 1

    flask_logger.info("New agent added to the session")
    flask_logger.debug(f"Updated session state: {session['state']}")
    save_current_session(session['state'])

@app.before_request
def before_request():
    flask_logger.debug(f"Before request for endpoint: {request.endpoint}")
    if request.endpoint in ['index', 'submit', 'get_past_chats']:
        check_session_state("before_request")

@app.route('/')
def index():
    if 'state' in session:
        return render_template('index.html', session_id=session['state']['session_id'])
    return render_template('index.html', session_id=None)

@app.route('/get_session_id', methods=['GET'])
def get_session_id():
    check_session_state("get_session_id")
    return jsonify({"session_id": session['state']['session_id']})

@app.route('/debug_session')
def debug_session():
    return jsonify(session.get('state', {}))

@app.route('/check_session')
def check_session():
    return jsonify({
        'session_id': session.get('state', {}).get('session_id'),
        'session_contents': session.get('state', {})
    })

@app.route('/update_session', methods=['POST'])
def update_session():
    try:
        new_session_id = request.json['session_id']
        loaded_state = load_state(new_session_id)
        if loaded_state:
            session['state'] = loaded_state
            return jsonify({"status": "success", "message": "Session updated successfully", "new_session_id": new_session_id}), 200
        return jsonify({"status": "error", "message": "Session not found"}), 404
    except Exception as e:
        flask_logger.error(f"Error updating session: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to update session"}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/get_agents', methods=['GET'])
def get_agents():
    if 'state' not in session:
        return jsonify({"error": "No active session"}), 400

    session_state = session['state']
    agents_df = pd.DataFrame(session_state['agents_df'])

    agents = [{"id": row['agent_id'], "name": row['agent_name']} for _, row in agents_df.iterrows()]
    return jsonify(agents)

@app.route('/get_past_chats')
def get_past_chats():
    if 'state' not in session:
        return jsonify({"error": "No active session"}), 400

    past_chats = []
    current_chat_id = session['state']['session_id']
    for filename in os.listdir(cache_dir):
        if filename.endswith('_state.json'):
            file_path = os.path.join(cache_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    state = json.load(f)
                    chat_id = filename.split('_')[0]
                    if chat_id == 'default':
                        continue  # Skip default settings file
                        
                    last_interaction = os.path.getmtime(file_path)

                    # Extract agent names from the state
                    agent_names = [agent['agent_name'] for agent in state.get('agents_df', [])]

                    past_chats.append({
                        'id': chat_id,
                        'user_name': state.get('user_name', 'User'),
                        'agent_names': agent_names,
                        'last_interaction': last_interaction,
                        'last_interaction_formatted': datetime.fromtimestamp(last_interaction).strftime('%Y-%m-%d %H:%M:%S'),
                        'is_current': chat_id == current_chat_id
                    })
            except Exception as e:
                flask_logger.error(f"Error loading chat file {filename}: {str(e)}")
                
    past_chats.sort(key=lambda x: x['last_interaction'], reverse=True)
    return jsonify(pastChats=past_chats)

@app.route('/load_chat/<chat_id>')
def load_chat(chat_id):
    session['state']['session_id'] = chat_id  # Update the current session ID
    state_file = os.path.join(cache_dir, f'{chat_id}_state.json')
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)

        agent_data = state['agents_df']
        agent_id_to_name = {agent['agent_id']: agent['agent_name'] for agent in agent_data}

        def get_speaker_name(speaker_id):
            if speaker_id == 'user':
                return state['user_name']
            return agent_id_to_name.get(speaker_id, 'Unknown')

        chat_html = ''
        for entry in state['session_history']:
            speaker_id, message = entry
            speaker_name = get_speaker_name(speaker_id)
            if speaker_id == 'user':
                chat_html += f'<div class="message user"><p>{message}</p></div>'
            else:
                chat_html += f'<div class="message agent"><p><strong>{speaker_name}:</strong> {message}</p></div>'

        agent_names = [agent['agent_name'] for agent in agent_data]
        agent_descriptions = [agent['description'] for agent in agent_data]
        agent_goals = [agent['goal'] for agent in agent_data]
        
        # Update session state with loaded state
        session['state'] = state
        save_current_session(state)  # Update last interaction time

        # Return needed info for frontend
        return jsonify({
            'history': state['session_history'],
            'chatHtml': chat_html,
            'agentNames': agent_names,
            'agentDescriptions': agent_descriptions,
            'agentGoals': agent_goals,
            'settings': state['settings'],
            'userSettings': {
                'user_name': state['user_name']
            }
        })
        
    except Exception as e:
        flask_logger.error(f"Error loading chat {chat_id}: {str(e)}")
        return jsonify({"error": "Failed to load chat"}), 500

@app.route('/update_last_interaction', methods=['POST'])
def update_last_interaction():
    try:
        chat_id = request.form.get('chat_id')
        if not chat_id:
            return jsonify({"error": "No chat ID provided"}), 400
            
        state_file = os.path.join(cache_dir, f'{chat_id}_state.json')
        if os.path.exists(state_file):
            # Read the file
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            # Write it back to update the modification time
            with open(state_file, 'w') as f:
                json.dump(state, f)
                
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Chat file not found"}), 404
    except Exception as e:
        flask_logger.error(f"Error updating last interaction time: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/visualize_pyvis', methods=['GET'])
def visualize_pyvis_route():
    selected_agent_id = request.args.get('agent_id')
    selected_chat_id = request.args.get('chat_id')

    if 'state' not in session:
        return jsonify({"error": "No active session"}), 400

    session_state = session['state']
    agents_df = pd.DataFrame(session_state['agents_df'])

    if selected_agent_id:
        if selected_agent_id in agents_df['agent_id'].values:
            agent = agents_df[agents_df['agent_id'] == selected_agent_id].iloc[0]
            graph_file_path = agent['graph_file_path']
        else:
            return jsonify({"error": f"Agent not found: {selected_agent_id}"}), 404
    elif selected_chat_id:
        # Load the chat state to get agent graphs
        chat_state_file = os.path.join(cache_dir, f'{selected_chat_id}_state.json')
        if os.path.exists(chat_state_file):
            with open(chat_state_file, 'r') as f:
                chat_state = json.load(f)
                
            chat_agents_df = pd.DataFrame(chat_state['agents_df'])
            if len(chat_agents_df) > 0:
                graph_file_path = chat_agents_df.iloc[0]['graph_file_path']
            else:
                return jsonify({"error": "No agents found in selected chat"}), 404
        else:
            return jsonify({"error": f"Chat state file not found: {chat_state_file}"}), 404
    else:
        # If no agent is selected, use the first agent's graph
        if len(agents_df) > 0:
            graph_file_path = agents_df.iloc[0]['graph_file_path']
        else:
            return jsonify({"error": "No agents available"}), 404

    # Ensure the graph file exists
    if not os.path.exists(graph_file_path):
        # Create an empty graph file
        graph = nx.DiGraph()
        graph.add_node('start')
        nx.write_graphml(graph, graph_file_path)

    # Generate visualization
    try:
        session_id = session_state['session_id']
        output_filename = visualize_graph_pyvis(graph_file_path, session_id)
        if output_filename:
            app.logger.debug(f'Visualizing graph with PyVis for {selected_agent_id or selected_chat_id or "default"}')
            return jsonify({"graph_html": f"/static/{output_filename}"})
        else:
            return jsonify({"error": "Failed to generate graph visualization"}), 500
    except Exception as e:
        app.logger.error(f"Error visualizing graph: {str(e)}")
        return jsonify({"error": "Failed to visualize graph"}), 500

@app.route('/save_agent_settings', methods=['POST'])
def save_agent_settings():
    if 'state' not in session:
        return jsonify({"error": "No active session"}), 400
        
    try:
        agent_id = request.form.get('agent_id')
        agent_name = request.form.get('agent_name')
        description = request.form.get('description')
        goal = request.form.get('goal')
        muted = request.form.get('muted') == 'true'
        use_agent_vars = request.form.get('use_agent_generation_variables') == 'true'
        
        # Generation parameters
        gen_vars = {
            'seed': int(request.form.get('seed', 42)),
            'temperature': float(request.form.get('temperature', 0.7)),
            'max_tokens': int(request.form.get('max_tokens', 150)),
            'top_p': float(request.form.get('top_p', 0.9)),
            'use_gpu': request.form.get('use_gpu') == 'true',
            'llm': None
        }
        
        # Update agent in session
        agents_df = pd.DataFrame(session['state']['agents_df'])
        if agent_id and agent_id in agents_df['agent_id'].values:
            idx = agents_df[agents_df['agent_id'] == agent_id].index[0]
            agents_df.at[idx, 'agent_name'] = agent_name
            agents_df.at[idx, 'description'] = description
            agents_df.at[idx, 'goal'] = goal
            agents_df.at[idx, 'muted'] = muted
            agents_df.at[idx, 'is_agent_generation_variables'] = use_agent_vars
            agents_df.at[idx, 'generation_variables'] = gen_vars
        else:
            # Add new agent
            new_agent = {
                'agent_id': generate_agent_id(),
                'agent_name': agent_name,
                'description': description,
                'goal': goal,
                'muted': muted,
                'environment': 'default',
                'graph_file_path': f"agent_{generate_agent_id()}_graph.graphml",
                'persistance_count': 0,
                'persistance_score': None,
                'patience': 6,
                'persistance': 3,
                'last_response': '',
                'last_narration': '',
                'current_aim': None,
                'suggestion': '',
                'current_node_location': 'start',
                'personal_history': [],
                'is_agent_generation_variables': use_agent_vars,
                'generation_variables': gen_vars,
                'impression_of_others': '',
                'environment_changes': '',
                'new_information': ''
            }
            
            # Initialize agent graph
            graph = nx.DiGraph()
            graph.add_node('start')
            nx.write_graphml(graph, os.path.join(graph_directory, new_agent['graph_file_path']))
            
            agents_df = pd.concat([agents_df, pd.DataFrame([new_agent])], ignore_index=True)
            session['state']['agent_mutes'].append(False)
            session['state']['number_of_agents'] += 1
        
        session['state']['agents_df'] = agents_df.to_dict('records')
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "message": "Agent settings saved",
            "agents": [{"id": row['agent_id'], "name": row['agent_name']} for _, row in agents_df.iterrows()]
        })
        
    except Exception as e:
        flask_logger.error(f"Error saving agent settings: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/delete_agent', methods=['POST'])
def delete_agent():
    if 'state' not in session:
        return jsonify({"error": "No active session"}), 400
        
    agent_id = request.form.get('agent_id')
    session_state = session['state']
    agents_df = pd.DataFrame(session_state['agents_df'])

    if agent_id in agents_df['agent_id'].values:
        # Get the index of the agent to delete
        agent_index = agents_df[agents_df['agent_id'] == agent_id].index[0]
        agent = agents_df.loc[agent_index]

        # Delete graph file
        graph_file = agent['graph_file_path']
        if os.path.exists(graph_file):
            os.remove(graph_file)

        # Remove agent from DataFrame
        agents_df = agents_df.drop(agent_index).reset_index(drop=True)
        session_state['agents_df'] = agents_df.to_dict('records')
        session_state['number_of_agents'] -= 1

        # Remove corresponding mute status
        if len(session_state['agent_mutes']) > agent_index:
            session_state['agent_mutes'].pop(agent_index)

        save_current_session(session_state)

        return jsonify({
            "success": True, 
            "message": "Agent deleted successfully",
            "agents": [{"id": row['agent_id'], "name": row['agent_name']} for _, row in agents_df.iterrows()]
        })
    else:
        return jsonify({"success": False, "message": "Agent not found"}), 404

@app.route('/submit', methods=['POST'])
def submit():
    flask_logger.info("Entering submit function")
    try:
        # Load the agent module
        agent_module = reload_agent()
        main = agent_module.offline_main if offline else agent_module.main
        flask_logger.info("Agent module reloaded successfully")

        check_session_state("submit")

        # If play is true, interrupt the current generation
        if session['state'].get('play', False):
            session['state']['play'] = False
            flask_logger.info("Play was true, setting to false and returning")
            return jsonify({
                "success": True,
                "history": session['state']['session_history'],
                "play": False
            })

        # Get user message and is_user flag from client side
        user_message = request.form.get('user_message', '')
        is_user = request.form.get('is_user') == 'true'
        flask_logger.info(f"Received user message: '{user_message}', is_user: {is_user}")

        # Record user message in history if is_user is true
        if is_user and user_message:
            session['state']['session_history'].append(("user", str(user_message)))
            session['state']['is_user'] = is_user
            session['state']['user_message'] = user_message
            flask_logger.info(f"Added user message to session history: '{user_message}'")
        else:
            flask_logger.info("No user message added to session history")

        # Get the max_generations value based on the is_user flag
        max_generations = len(session['state']['agents_df']) if is_user else 100
        flask_logger.info(f"Set max_generations to {max_generations}")

        # Initialize the generation process
        session['state']['current_generation'] = 0
        session['state']['max_generations'] = max_generations
        session['state']['play'] = True
        flask_logger.info("Initialized generation process variables")
        
        # Save the session state
        save_current_session(session['state'])

        response = jsonify({
            "success": True,
            "max_generations": max_generations,
            "history": session['state']['session_history'],
            "play": session['state']['play']
        })
        flask_logger.info(f"Submit response prepared: {response.get_data(as_text=True)}")
        flask_logger.info("Exiting submit function successfully")
        return response

    except Exception as e:
        flask_logger.error(f"Error in submit function: {str(e)}", exc_info=True)
        return jsonify({
            "success": False, 
            "error": f"Error in submit function: {str(e)}"
        }), 500

@app.route('/generate', methods=['GET'])
def generate():
    flask_logger.info("Entering generate function")
    try:
        # Check if there's an active session
        if 'state' not in session:
            flask_logger.error("No active session in generate function")
            return jsonify({"error": "No active session"}), 400

        # Load the agent module
        agent_module = reload_agent()
        main = agent_module.offline_main if offline else agent_module.main

        # Check if generation should continue
        flask_logger.info(f"Current play status: {session['state']['play']}")
        if not session['state']['play']:
            flask_logger.info("Generation stopped due to play status")
            return jsonify({"complete": True, "play": False})

        # Check if max generations reached
        flask_logger.info(f"Current generation: {session['state']['current_generation']}, Max generations: {session['state']['max_generations']}")
        if session['state']['current_generation'] >= session['state']['max_generations']:
            flask_logger.info("Generation complete")
            session['state']['play'] = False
            save_current_session(session['state'])
            return jsonify({"complete": True, "play": False})

        # Prep variables for main function
        is_user = session['state']['is_user']
        agents_df = pd.DataFrame(session['state']['agents_df'])
        settings = session['state']['settings']
        user_name = session['state']['user_name']
        agent_mutes = session['state'].get('agent_mutes', [])
        len_last_history = session['state'].get('len_last_history', 0)

        # Call the main function
        flask_logger.info(f"Calling main function with: is_user={is_user}, user_name={user_name}, agent_mutes={agent_mutes}, len_last_history={len_last_history}")
        new_history, new_agents_df, logs = main(
            session['state']['session_history'],
            agents_df,
            settings,
            user_name,
            is_user,
            agent_mutes,
            len_last_history
        )

        # Check if new response was generated
        if len(new_history) > len(session['state']['session_history']):
            latest_response = new_history[-1]
            flask_logger.info(f"New response generated: {latest_response}")
        else:
            flask_logger.info("No new response generated")

        # Update session state
        session['state']['session_history'] = new_history
        session['state']['len_last_history'] = len(new_history)
        session['state']['agents_df'] = new_agents_df.to_dict('records')
        session['state']['current_generation'] += 1

        # Determine if generation should continue
        continue_generation = session['state']['current_generation'] < session['state']['max_generations']
        session['state']['play'] = continue_generation
        flask_logger.info(f"Updated play status: {session['state']['play']}")

        # Save the updated session
        save_current_session(session['state'])

        # Prepare and return response
        response = jsonify({
            "history": session['state']['session_history'],
            "logs": logs,
            "play": session['state']['play'],
            "clear_chatbox": True,
            "current_generation": session['state']['current_generation'],
            "max_generations": session['state']['max_generations']
        })
        flask_logger.info("Exiting generate function successfully")
        return response

    except Exception as e:
        flask_logger.error(f"Error in generate function: {str(e)}", exc_info=True)
        session['state']['play'] = False
        save_current_session(session['state'])
        return jsonify({
            "error": str(e), 
            "play": False
        }), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    """
    Reset the current chat - clears history and environment, but keeps agent settings.
    This follows the 'restart option' in the TODO list.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        # Keep the existing session ID and agent settings
        session_id = session['state']['session_id']
        agents_df = pd.DataFrame(session['state']['agents_df'])
        settings = session['state']['settings']
        user_name = session['state']['user_name']
        
        # Reset dynamic conversation elements
        for idx, agent in agents_df.iterrows():
            agents_df.at[idx, 'current_aim'] = None
            agents_df.at[idx, 'suggestion'] = ''
            agents_df.at[idx, 'persistance_count'] = 0
            agents_df.at[idx, 'persistance_score'] = None
            agents_df.at[idx, 'last_response'] = ''
            agents_df.at[idx, 'last_narration'] = ''
            agents_df.at[idx, 'personal_history'] = []
            agents_df.at[idx, 'current_node_location'] = 'start'
            agents_df.at[idx, 'impression_of_others'] = ''
            agents_df.at[idx, 'environment_changes'] = ''
            agents_df.at[idx, 'new_information'] = ''
        
        # Reset session state but keep settings
        session['state'] = {
            'session_id': session_id,
            'user_name': user_name,
            'agent_mutes': [False] * len(agents_df),
            'agents_df': agents_df.to_dict('records'),
            'session_history': [],
            'len_last_history': 0,
            'settings': settings,
            'play': False,
            'is_user': False,
            'max_generations': 0,
            'current_generation': 0,
            'user_message': '',
            'number_of_agents': len(agents_df)
        }
        
        # Save the reset state
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "message": "Chat reset successfully"
        })
    except Exception as e:
        flask_logger.error(f"Error resetting chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/duplicate', methods=['POST'])
def duplicate_chat():
    """
    Duplicate the current chat - creates a new chat with the same state.
    This follows the 'duplicate chat option' in the TODO list.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        # Copy the current state
        current_state = session['state'].copy()
        
        # Generate a new session ID
        new_session_id = generate_session_id()
        current_state['session_id'] = new_session_id
        
        # Save as a new chat
        save_current_session(current_state)
        
        # Update the session to use the new chat
        session['state'] = current_state
        
        return jsonify({
            "success": True,
            "message": "Chat duplicated successfully",
            "new_session_id": new_session_id
        })
    except Exception as e:
        flask_logger.error(f"Error duplicating chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/create_new', methods=['POST'])
def create_new_chat():
    """
    Create a new chat - loads defaults and changes to settings page.
    This follows the 'create new chat' option in the TODO list.
    """
    try:
        # Initialize a new session with defaults
        new_session = initialize_session()
        session['state'] = new_session
        
        return jsonify({
            "success": True,
            "message": "New chat created",
            "session_id": new_session['session_id'],
            "redirect_to": "agent"  # Redirect to settings page
        })
    except Exception as e:
        flask_logger.error(f"Error creating new chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    """
    Delete the current chat from cache and create a new one.
    This follows the 'delete chat option' in the TODO list.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        # Get the current session ID
        session_id = session['state']['session_id']
        
        # Delete the cache file
        state_file = os.path.join(cache_dir, f'{session_id}_state.json')
        if os.path.exists(state_file):
            os.remove(state_file)
            
        # Create a new session
        new_session = initialize_session()
        session['state'] = new_session
        
        return jsonify({
            "success": True,
            "message": "Chat deleted and new chat created",
            "session_id": new_session['session_id'],
            "redirect_to": "agent"  # Redirect to settings page
        })
    except Exception as e:
        flask_logger.error(f"Error deleting chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/rename_chat', methods=['POST'])
def rename_chat():
    """
    Rename the current chat.
    This handles the chat renaming functionality mentioned in the TODO list.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        # Get the new name from the request
        new_name = request.form.get('new_name')
        if not new_name:
            return jsonify({
                "success": False,
                "error": "No name provided"
            }), 400
            
        # Update the chat name
        session['state']['chat_name'] = new_name
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "message": "Chat renamed successfully",
            "new_name": new_name
        })
    except Exception as e:
        flask_logger.error(f"Error renaming chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/update_user_settings', methods=['POST'])
def update_user_settings():
    """
    Update user settings for the current session.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        user_name = request.form.get('user_name')
        if user_name:
            session['state']['user_name'] = user_name
            
        # Update generation settings if provided
        for param in ['temperature', 'max_tokens', 'top_p', 'seed']:
            if param in request.form:
                try:
                    value = request.form.get(param)
                    if param in ['temperature', 'top_p']:
                        session['state']['settings'][param] = float(value)
                    else:
                        session['state']['settings'][param] = int(value)
                except ValueError:
                    flask_logger.warning(f"Invalid value for {param}: {value}")
        
        # Update use_gpu setting
        use_gpu = request.form.get('use_gpu')
        if use_gpu is not None:
            session['state']['settings']['use_gpu'] = use_gpu.lower() == 'true'
            
        # Save the updated session
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "message": "User settings updated successfully",
            "settings": session['state']['settings'],
            "user_name": session['state']['user_name']
        })
    except Exception as e:
        flask_logger.error(f"Error updating user settings: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/set_chat_mode', methods=['POST'])
def set_chat_mode():
    """
    Set the chat mode to either user-agent or agent-agent.
    This implements the functionality for "User x Agent" OR "Agent x Agent" option.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        mode = request.form.get('mode')
        if not mode or mode not in ['user_agent', 'agent_agent']:
            return jsonify({
                "success": False,
                "error": "Invalid chat mode specified"
            }), 400
            
        # Update the chat mode
        session['state']['chat_mode'] = mode
        
        # For agent-agent mode, ensure we have at least two agents
        if mode == 'agent_agent':
            agents_df = pd.DataFrame(session['state']['agents_df'])
            if len(agents_df) < 2:
                # Add a second agent if needed
                default_agent = load_default_settings()['agents_df'][1]  # Get the second default agent
                add_single_agent_to_session(default_agent)
                
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "message": f"Chat mode set to {mode}",
            "chat_mode": mode,
            "agent_count": session['state']['number_of_agents']
        })
    except Exception as e:
        flask_logger.error(f"Error setting chat mode: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/toggle_agent_mute', methods=['POST'])
def toggle_agent_mute():
    """
    Toggle mute status for a specific agent
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        agent_id = request.form.get('agent_id')
        if not agent_id:
            return jsonify({
                "success": False,
                "error": "No agent ID provided"
            }), 400
            
        # Find the agent
        agents_df = pd.DataFrame(session['state']['agents_df'])
        if agent_id not in agents_df['agent_id'].values:
            return jsonify({
                "success": False,
                "error": "Agent not found"
            }), 404
            
        # Get the agent index
        agent_index = agents_df[agents_df['agent_id'] == agent_id].index[0]
        
        # Toggle mute status
        agents_df.at[agent_index, 'muted'] = not agents_df.at[agent_index, 'muted']
        session['state']['agents_df'] = agents_df.to_dict('records')
        
        # Update agent_mutes array to match
        session['state']['agent_mutes'][agent_index] = agents_df.at[agent_index, 'muted']
        
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "agent_id": agent_id,
            "muted": agents_df.at[agent_index, 'muted']
        })
    except Exception as e:
        flask_logger.error(f"Error toggling agent mute status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/respond_as_agent', methods=['POST'])
def respond_as_agent():
    """
    Add a response to the chat as if it came from a specific agent.
    This enables manual interaction in agent-agent mode.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        agent_id = request.form.get('agent_id')
        message = request.form.get('message')
        
        if not agent_id or not message:
            return jsonify({
                "success": False,
                "error": "Agent ID and message are required"
            }), 400
            
        # Find the agent
        agents_df = pd.DataFrame(session['state']['agents_df'])
        if agent_id not in agents_df['agent_id'].values:
            return jsonify({
                "success": False,
                "error": "Agent not found"
            }), 404
            
        # Add the message to the history
        session['state']['session_history'].append((agent_id, message))
        session['state']['len_last_history'] = len(session['state']['session_history'])
        
        # Update agent personal histories
        for idx, agent in agents_df.iterrows():
            if not agent['muted']:
                personal_history = agent['personal_history']
                personal_history.append((agent_id, message))
                agents_df.at[idx, 'personal_history'] = personal_history
                
        session['state']['agents_df'] = agents_df.to_dict('records')
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "history": session['state']['session_history']
        })
    except Exception as e:
        flask_logger.error(f"Error adding agent response: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/delete_last_response', methods=['POST'])
def delete_last_response():
    """
    Delete the last response in the chat history.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        if not session['state']['session_history']:
            return jsonify({
                "success": False,
                "error": "No messages to delete"
            }), 400
            
        # Remove the last message
        session['state']['session_history'].pop()
        session['state']['len_last_history'] = len(session['state']['session_history'])
        
        # Update agent personal histories
        agents_df = pd.DataFrame(session['state']['agents_df'])
        for idx, agent in agents_df.iterrows():
            personal_history = agent['personal_history']
            if personal_history:
                personal_history.pop()
                agents_df.at[idx, 'personal_history'] = personal_history
                
        session['state']['agents_df'] = agents_df.to_dict('records')
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "history": session['state']['session_history']
        })
    except Exception as e:
        flask_logger.error(f"Error deleting last response: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/update_generation_parameters', methods=['POST'])
def update_generation_parameters():
    """
    Update advanced generation parameters for the session or specific agent.
    This implements the "additional generation parameters" requirement.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        agent_id = request.form.get('agent_id')
        
        # Parameters to update
        params = {}
        for param in ['temperature', 'max_tokens', 'top_p', 'seed', 'top_k', 'repetition_penalty']:
            if param in request.form:
                try:
                    value = request.form.get(param)
                    if param in ['temperature', 'top_p', 'repetition_penalty']:
                        params[param] = float(value)
                    else:
                        params[param] = int(value)
                except ValueError:
                    flask_logger.warning(f"Invalid value for {param}: {value}")
        
        if 'use_gpu' in request.form:
            params['use_gpu'] = request.form.get('use_gpu').lower() == 'true'
            
        # If agent_id is provided, update that specific agent
        if agent_id:
            agents_df = pd.DataFrame(session['state']['agents_df'])
            if agent_id in agents_df['agent_id'].values:
                idx = agents_df[agents_df['agent_id'] == agent_id].index[0]
                for param, value in params.items():
                    agents_df.at[idx, 'generation_variables'][param] = value
                session['state']['agents_df'] = agents_df.to_dict('records')
            else:
                return jsonify({
                    "success": False,
                    "error": "Agent not found"
                }), 404
        # Otherwise update session settings
        else:
            for param, value in params.items():
                session['state']['settings'][param] = value
        
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "message": f"Generation parameters updated for {'agent '+agent_id if agent_id else 'session'}"
        })
    except Exception as e:
        flask_logger.error(f"Error updating generation parameters: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/save_graph', methods=['POST'])
def save_graph():
    """
    Save the current graph to a file.
    This implements the "option to save down Graph" requirement.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        agent_id = request.form.get('agent_id')
        if not agent_id:
            return jsonify({
                "success": False,
                "error": "No agent ID provided"
            }), 400
            
        # Find the agent
        agents_df = pd.DataFrame(session['state']['agents_df'])
        if agent_id not in agents_df['agent_id'].values:
            return jsonify({
                "success": False,
                "error": "Agent not found"
            }), 404
            
        agent = agents_df[agents_df['agent_id'] == agent_id].iloc[0]
        graph_file_path = agent['graph_file_path']
        
        if not os.path.exists(graph_file_path):
            return jsonify({
                "success": False,
                "error": "Graph file not found"
            }), 404
            
        # Generate the visualization
        session_id = session['state']['session_id']
        output_filename = visualize_graph_pyvis(graph_file_path, session_id)
        
        if not output_filename:
            return jsonify({
                "success": False,
                "error": "Failed to generate graph visualization"
            }), 500
            
        # Create a download URL
        download_url = f"/static/{output_filename}"
        
        return jsonify({
            "success": True,
            "download_url": download_url,
            "agent_name": agent['agent_name']
        })
    except Exception as e:
        flask_logger.error(f"Error saving graph: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/export_chat_json', methods=['POST'])
def export_chat_json():
    """
    Export the current chat state as a JSON file.
    This implements the "option to save down Json" requirement.
    """
    try:
        if 'state' not in session:
            return jsonify({"error": "No active session"}), 400
            
        # Create a copy of the state to sanitize
        export_state = session['state'].copy()
        
        # Remove any sensitive or unnecessary information
        if 'llm' in export_state.get('settings', {}):
            export_state['settings']['llm'] = None
            
        for agent in export_state.get('agents_df', []):
            if 'llm' in agent.get('generation_variables', {}):
                agent['generation_variables']['llm'] = None
        
        # Generate a filename based on agent and user names
        user_name = export_state.get('user_name', 'User').replace(' ', '_')
        agent_names = '_'.join([agent['agent_name'].replace(' ', '_') for agent in export_state.get('agents_df', [])])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{user_name}_{agent_names}_{timestamp}.json"
        
        # Save the file
        export_path = os.path.join(app.static_folder, filename)
        with open(export_path, 'w') as f:
            json.dump(export_state, f, indent=2)
        
        # Create a download URL
        download_url = f"/static/{filename}"
        
        return jsonify({
            "success": True,
            "download_url": download_url,
            "filename": filename
        })
    except Exception as e:
        flask_logger.error(f"Error exporting chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/interrupt', methods=['POST'])
def interrupt():
    """
    Interrupt the current generation process.
    """
    if 'state' in session:
        session['state']['play'] = False
        save_current_session(session['state'])
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "No active session"}), 400

@app.errorhandler(404)
def not_found_error(error):
    flask_logger.error(f'404 error: {request.url}')
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    flask_logger.error(f'500 error: {error}')
    return jsonify({"error": "Internal server error"}), 500

# Add the main execution block to run the app
if __name__ == '__main__':
    import sys
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    
    # Check if model exists or download it
    if not os.path.exists(model_filename):
        try:
            download_model()
        except Exception as e:
            flask_logger.error(f"Failed to download model: {str(e)}")
            print(f"Failed to download model: {str(e)}")
            sys.exit(1)
            
    # Load a test graph to ensure the graph directory is correctly set up
    test_graph = nx.DiGraph()
    test_graph.add_node('start')
    os.makedirs(graph_directory, exist_ok=True)
    test_graph_path = os.path.join(graph_directory, 'test_graph.graphml')
    nx.write_graphml(test_graph, test_graph_path)
    
    flask_logger.info(f"Starting Flask app on port {port}")
    from waitress import serve
    serve(app, host='0.0.0.0', port=port)

@app.route('/add_agent', methods=['POST'])
def add_agent():
    if 'state' not in session:
        return jsonify({"error": "No active session"}), 400
        
    try:
        # Create a new agent with default settings
        agent_id = generate_agent_id()
        new_agent = {
            'agent_id': agent_id,
            'agent_name': 'New Agent',
            'muted': False,
            'description': 'A new intelligent agent',
            'environment': 'default',
            'graph_file_path': os.path.join(graph_directory, f"{agent_id}_graph.graphml"),
            'persistance_count': 0,
            'persistance_score': None,
            'patience': 6,
            'persistance': 3,
            'last_response': '',
            'last_narration': '',
            'goal': 'Complete a task',
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
        
        # Initialize agent graph
        graph = nx.DiGraph()
        graph.add_node('start')
        nx.write_graphml(graph, new_agent['graph_file_path'])
        
        # Add the new agent to the session
        agents_df = pd.DataFrame(session['state']['agents_df'])
        agents_df = pd.concat([agents_df, pd.DataFrame([new_agent])], ignore_index=True)
        session['state']['agents_df'] = agents_df.to_dict('records')
        session['state']['agent_mutes'].append(False)
        session['state']['number_of_agents'] += 1
        
        save_current_session(session['state'])
        
        return jsonify({
            "success": True,
            "message": "New agent added",
            "agent": {
                "id": new_agent['agent_id'],
                "name": new_agent['agent_name']
            }
        })
    except Exception as e:
        flask_logger.error(f"Error adding new agent: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500