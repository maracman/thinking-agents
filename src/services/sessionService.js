/**
 * Service for handling session-related operations
 */

import { checkSession, updateSession } from './api';

/**
 * Check if a session exists and is valid
 * @param {string} sessionId - The session ID to check
 * @returns {Promise<boolean>} Whether the session is valid
 */
export const isSessionValid = async (sessionId) => {
  if (!sessionId) return false;
  
  try {
    const response = await checkSession();
    return response.session_id === sessionId && response.session_contents;
  } catch (error) {
    console.error("Error checking session validity:", error);
    return false;
  }
};

/**
 * Format session history for display
 * @param {Array} history - Raw session history
 * @param {Object} agentMap - Map of agent IDs to names
 * @param {string} userName - User's display name
 * @returns {Array} Formatted history items with display names
 */
export const formatSessionHistory = (history, agentMap, userName) => {
  if (!history || !Array.isArray(history)) {
    return [];
  }
  
  return history.map(([speakerId, message]) => {
    const displayName = speakerId === 'user' 
      ? userName 
      : agentMap[speakerId] || speakerId;
    
    return {
      id: Math.random().toString(36).substr(2, 9), // Simple unique ID
      speakerId,
      displayName,
      message,
      timestamp: Date.now() // Or use a provided timestamp if available
    };
  });
};

/**
 * Create agent ID to name mapping
 * @param {Array} agents - List of agent objects
 * @returns {Object} Map of agent IDs to names
 */
export const createAgentMap = (agents) => {
  if (!agents || !Array.isArray(agents)) {
    return {};
  }
  
  return agents.reduce((map, agent) => {
    if (agent && agent.agent_id) {
      map[agent.agent_id] = agent.agent_name || 'Unknown Agent';
    }
    return map;
  }, {});
};

/**
 * Get default settings for a new session
 * @returns {Object} Default session settings
 */
export const getDefaultSettings = () => {
  return {
    temperature: 0.7,
    max_tokens: 150,
    top_p: 0.9,
    seed: 42,
    top_k: 40,
    repetition_penalty: 1.1,
    use_gpu: true
  };
};

/**
 * Save session state to local storage (for quick recovery)
 * @param {string} sessionId - Session ID
 * @param {Object} state - Session state to save
 */
export const saveStateToLocalStorage = (sessionId, state) => {
  if (!sessionId || !state) return;
  
  try {
    const key = `session_${sessionId}`;
    const stateString = JSON.stringify({
      timestamp: Date.now(),
      state: {
        ...state,
        // Omit any large data or sensitive information
        history: state.history?.slice(-20) || [] // Only save recent history
      }
    });
    
    localStorage.setItem(key, stateString);
  } catch (error) {
    console.error("Error saving state to local storage:", error);
  }
};

/**
 * Load session state from local storage
 * @param {string} sessionId - Session ID
 * @returns {Object|null} Saved session state or null if not found
 */
export const loadStateFromLocalStorage = (sessionId) => {
  if (!sessionId) return null;
  
  try {
    const key = `session_${sessionId}`;
    const stateString = localStorage.getItem(key);
    
    if (!stateString) return null;
    
    const data = JSON.parse(stateString);
    
    // Check if data is too old (more than 1 hour)
    const isExpired = Date.now() - data.timestamp > 3600000;
    
    if (isExpired) {
      localStorage.removeItem(key);
      return null;
    }
    
    return data.state;
  } catch (error) {
    console.error("Error loading state from local storage:", error);
    return null;
  }
};

/**
 * Compare two session states and check for significant changes
 * @param {Object} prevState - Previous session state
 * @param {Object} newState - New session state
 * @returns {boolean} Whether significant changes exist
 */
export const hasSessionChanged = (prevState, newState) => {
  if (!prevState || !newState) return true;
  
  // Check history length
  if (prevState.history?.length !== newState.history?.length) {
    return true;
  }
  
  // Check agent count
  if (prevState.agentData?.length !== newState.agentData?.length) {
    return true;
  }
  
  // Check play state
  if (prevState.isPlaying !== newState.isPlaying) {
    return true;
  }
  
  return false;
};

export default {
  isSessionValid,
  formatSessionHistory,
  createAgentMap,
  getDefaultSettings,
  saveStateToLocalStorage,
  loadStateFromLocalStorage,
  hasSessionChanged
};