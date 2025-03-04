import React, { useState, useEffect } from 'react';
import Sidebar from './Sidebar';
import ChatInterface from './ChatInterface';
import AgentSettings from './AgentSettings';
import GraphView from './GraphView';
import DeveloperTools from './DeveloperTools';
import { fetchSessionId, checkSession } from '../services/api';

const App = () => {
  const [activeTab, setActiveTab] = useState('chat');
  const [sessionId, setSessionId] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sessionState, setSessionState] = useState({
    agentData: [],
    history: [],
    settings: {},
    isPlaying: false,
    isUser: false,
    userMessage: '',
    maxGenerations: 0,
    currentGeneration: 0
  });
  
  useEffect(() => {
    // Initialize session on component mount
    const initSession = async () => {
      try {
        const id = await fetchSessionId();
        setSessionId(id);
        
        const sessionData = await checkSession();
        if (sessionData.session_contents) {
          setSessionState(prevState => ({
            ...prevState,
            isUser: sessionData.session_contents.is_user,
            isPlaying: sessionData.session_contents.play
          }));
        }
      } catch (error) {
        console.error("Error initializing session:", error);
      }
    };
    
    initSession();
  }, []);
  
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
        sessionState={sessionState}
      />
      
      <main className={`main-content ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        {activeTab === 'chat' && (
          <ChatInterface 
            sessionId={sessionId} 
            sessionState={sessionState}
            setSessionState={setSessionState}
          />
        )}
        
        {activeTab === 'agent' && (
          <AgentSettings 
            sessionId={sessionId}
            sessionState={sessionState}
            setSessionState={setSessionState}
          />
        )}
        
        {activeTab === 'graph' && (
          <GraphView 
            sessionId={sessionId}
            agentData={sessionState.agentData}
          />
        )}
        
        {activeTab === 'developer' && (
          <DeveloperTools 
            sessionId={sessionId}
          />
        )}
      </main>
    </div>
  );
};

export default App;