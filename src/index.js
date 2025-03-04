import React from 'react';
import ReactDOM from 'react-dom';
import App from './components/App';
import { SessionProvider } from './contexts/SessionContext';
import { AgentProvider } from './contexts/AgentContext';
import './styles/main.css';

// Grab initial data passed from the backend template
const initialData = window.initialData || {};

// Initialize LLM provider from localStorage if available
const initializeLLM = () => {
  // Import LLM service
  const llmService = require('./services/llmService').default;
  const { setProvider, setApiKey } = require('./services/llmService');
  
  // Get saved LLM settings
  const savedProvider = localStorage.getItem('llm_provider');
  const savedModel = localStorage.getItem('llm_selected_model');
  
  // Set provider if saved
  if (savedProvider) {
    setProvider(savedProvider);
    
    // For local provider, no API key is needed
    if (savedProvider !== 'local') {
      // In a real app, you would retrieve API keys from a secure source
      // This is just a placeholder - in production, never store API keys in localStorage
      const apiKey = localStorage.getItem(`${savedProvider}_api_key`);
      if (apiKey) {
        setApiKey(savedProvider, apiKey);
      }
    }
  }
  
  return llmService;
};

// Initialize LLM provider
initializeLLM();

ReactDOM.render(
  <React.StrictMode>
    <SessionProvider initialSessionId={initialData.sessionId}>
      <AgentProvider>
        <App />
      </AgentProvider>
    </SessionProvider>
  </React.StrictMode>,
  document.getElementById('root')
);