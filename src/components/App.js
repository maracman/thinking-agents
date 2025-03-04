import React, { useState } from 'react';
import Sidebar from './Sidebar';
import ChatInterface from './ChatInterface';
import AgentSettings from './AgentSettings';
import GraphView from './GraphView';
import DeveloperTools from './DeveloperTools';
import LLMSettings from './LLMSettings';
import { useSession } from '../contexts/SessionContext';
import { useAgent } from '../contexts/AgentContext';

const App = () => {
  // Use context instead of local state
  const { sessionId, sessionState } = useSession();
  const { agents } = useAgent();
  
  // Local UI state only
  const [activeTab, setActiveTab] = useState('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };
  
  return (
    <div className="app-container">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        collapsed={sidebarCollapsed}
        toggleSidebar={toggleSidebar}
      />
      
      <main className={`main-content ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        {activeTab === 'chat' && (
          <ChatInterface />
        )}
        
        {activeTab === 'agent' && (
          <AgentSettings />
        )}
        
        {activeTab === 'graph' && (
          <GraphView />
        )}
        
        {activeTab === 'developer' && (
          <DeveloperTools />
        )}
        
        {activeTab === 'llm' && (
          <LLMSettings />
        )}
      </main>
    </div>
  );
};

export default App;