import React, { createContext, useState, useEffect, useContext } from 'react';
import { useSession } from './SessionContext';
import { 
  fetchAgents, 
  saveAgentSettings, 
  deleteAgent, 
  addNewAgent,
  fetchAgentGraphs,
  visualizeGraph,
  toggleAgentMute
} from '../services/api';

// Create context
const AgentContext = createContext();

export const AgentProvider = ({ children }) => {
  const { sessionId, sessionState, setSessionState } = useSession();
  
  // Agent state
  const [agents, setAgents] = useState([]);
  const [selectedAgentIndex, setSelectedAgentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Agent form data
  const [formData, setFormData] = useState({
    agent_name: '',
    agent_description: '',
    goal: '',
    target_impression: '',
    muted: false,
    use_agent_generation_variables: false,
    generation_variables: {
      seed: 42,
      temperature: 0.7,
      max_tokens: 150,
      top_p: 0.9,
      use_gpu: true
    }
  });
  
  // Graph state
  const [agentGraphs, setAgentGraphs] = useState([]);
  const [selectedGraphAgentId, setSelectedGraphAgentId] = useState('');
  const [graphUrl, setGraphUrl] = useState('');
  const [graphLoading, setGraphLoading] = useState(false);
  
  // Load agents when sessionId changes
  useEffect(() => {
    if (sessionId) {
      loadAgents();
    }
  }, [sessionId]);
  
  // Update form data when selected agent changes
  useEffect(() => {
    if (agents.length > 0 && selectedAgentIndex < agents.length) {
      const agent = agents[selectedAgentIndex];
      setFormData({
        agent_name: agent.agent_name,
        agent_description: agent.description,
        goal: agent.goal,
        target_impression: agent.target_impression || '',
        muted: agent.muted,
        use_agent_generation_variables: agent.is_agent_generation_variables,
        generation_variables: {
          seed: agent.generation_variables?.seed || 42,
          temperature: agent.generation_variables?.temperature || 0.7,
          max_tokens: agent.generation_variables?.max_tokens || 150,
          top_p: agent.generation_variables?.top_p || 0.9,
          use_gpu: agent.generation_variables?.use_gpu || true
        }
      });
    }
  }, [selectedAgentIndex, agents]);
  
  // Load agents from API
  const loadAgents = async () => {
    try {
      setLoading(true);
      const agentsData = await fetchAgents();
      setAgents(agentsData);
      
      // Update global session state
      setSessionState(prev => ({
        ...prev,
        agentData: agentsData
      }));
      
      setError(null);
    } catch (err) {
      console.error("Error loading agents:", err);
      setError("Failed to load agents");
    } finally {
      setLoading(false);
    }
  };
  
  // Handle agent selection change
  const handleAgentChange = (index) => {
    setSelectedAgentIndex(Number(index));
  };
  
  // Handle form input change
  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };
  
  // Handle generation variable change
  const handleGenerationVariableChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : 
                    type === 'number' ? Number(value) : 
                    value;
    
    setFormData(prev => ({
      ...prev,
      generation_variables: {
        ...prev.generation_variables,
        [name]: newValue
      }
    }));
  };
  
  // Save agent settings
  const handleSaveSettings = async () => {
    try {
      setLoading(true);
      
      if (agents.length === 0 || selectedAgentIndex >= agents.length) {
        throw new Error("No agent selected");
      }
      
      const agentId = agents[selectedAgentIndex].agent_id;
      
      const updatedAgent = {
        ...agents[selectedAgentIndex],
        agent_name: formData.agent_name,
        description: formData.agent_description,
        goal: formData.goal,
        target_impression: formData.target_impression,
        muted: formData.muted,
        is_agent_generation_variables: formData.use_agent_generation_variables,
        generation_variables: formData.generation_variables
      };
      
      const response = await saveAgentSettings(agentId, updatedAgent);
      
      if (response.success) {
        // Refresh the agents list
        await loadAgents();
        return { success: true };
      } else {
        throw new Error(response.error || "Failed to save agent settings");
      }
    } catch (err) {
      console.error("Error saving agent settings:", err);
      setError(err.message || "Failed to save agent settings");
      return { success: false, error: err.message };
    } finally {
      setLoading(false);
    }
  };
  
  // Delete agent
  const handleDeleteAgent = async () => {
    try {
      setLoading(true);
      
      if (agents.length <= 1) {
        throw new Error("Cannot delete the last agent. At least one agent is required.");
      }
      
      const agentId = agents[selectedAgentIndex].agent_id;
      const response = await deleteAgent(agentId);
      
      if (response.success) {
        // Refresh the agents list and reset selection
        await loadAgents();
        setSelectedAgentIndex(0);
        return { success: true };
      } else {
        throw new Error(response.error || "Failed to delete agent");
      }
    } catch (err) {
      console.error("Error deleting agent:", err);
      setError(err.message || "Failed to delete agent");
      return { success: false, error: err.message };
    } finally {
      setLoading(false);
    }
  };
  
  // Add new agent
  const handleAddAgent = async () => {
    try {
      setLoading(true);
      
      const response = await addNewAgent();
      
      if (response.success) {
        // Refresh the agents list and select the new agent
        const updatedAgents = await fetchAgents();
        setAgents(updatedAgents);
        setSelectedAgentIndex(updatedAgents.length - 1);
        return { success: true };
      } else {
        throw new Error(response.error || "Failed to add new agent");
      }
    } catch (err) {
      console.error("Error adding new agent:", err);
      setError(err.message || "Failed to add new agent");
      return { success: false, error: err.message };
    } finally {
      setLoading(false);
    }
  };
  
  // Load agent graphs
  const loadAgentGraphs = async () => {
    try {
      setGraphLoading(true);
      const graphData = await fetchAgentGraphs();
      setAgentGraphs(graphData);
      
      // Select the first agent by default if there's no selection and agents are available
      if (!selectedGraphAgentId && graphData.length > 0) {
        setSelectedGraphAgentId(graphData[0].agent_id);
        await loadGraph(graphData[0].agent_id);
      }
      
      setError(null);
    } catch (err) {
      console.error("Error loading agent graphs:", err);
      setError("Failed to load agent graphs");
    } finally {
      setGraphLoading(false);
    }
  };
  
  // Handle graph agent selection change
  const handleGraphAgentChange = async (agentId) => {
    setSelectedGraphAgentId(agentId);
    await loadGraph(agentId);
  };
  
  // Load graph for specific agent
  const loadGraph = async (agentId) => {
    if (!agentId) return;
    
    setGraphLoading(true);
    try {
      const graphData = await visualizeGraph(agentId);
      setGraphUrl(graphData.graph_html);
      setError(null);
    } catch (err) {
      console.error("Error visualizing graph:", err);
      setError("Failed to visualize graph");
    } finally {
      setGraphLoading(false);
    }
  };
  
  // Toggle agent mute status
  const handleToggleAgentMute = async (agentId) => {
    try {
      const response = await toggleAgentMute(agentId);
      
      if (response.success) {
        // Refresh the agents list
        await loadAgents();
        return { success: true, muted: response.muted };
      } else {
        throw new Error(response.error || "Failed to toggle agent mute status");
      }
    } catch (err) {
      console.error("Error toggling agent mute status:", err);
      setError(err.message || "Failed to toggle agent mute status");
      return { success: false, error: err.message };
    }
  };
  
  // Value to be provided by the context
  const value = {
    agents,
    selectedAgentIndex,
    formData,
    loading,
    error,
    agentGraphs,
    selectedGraphAgentId,
    graphUrl,
    graphLoading,
    handleAgentChange,
    handleInputChange,
    handleGenerationVariableChange,
    handleSaveSettings,
    handleDeleteAgent,
    handleAddAgent,
    loadAgentGraphs,
    handleGraphAgentChange,
    loadGraph,
    handleToggleAgentMute
  };
  
  return (
    <AgentContext.Provider value={value}>
      {children}
    </AgentContext.Provider>
  );
};

// Custom hook for using the agent context
export const useAgent = () => {
  const context = useContext(AgentContext);
  if (!context) {
    throw new Error('useAgent must be used within an AgentProvider');
  }
  return context;
};

export default AgentContext;