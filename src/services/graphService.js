/**
 * Service for handling graph visualization and processing
 */

/**
 * Process raw graph data for visualization
 * @param {Array} graphData - Raw graph data from the API
 * @returns {Object} Processed graph data ready for visualization
 */
export const processGraphData = (graphData) => {
    if (!graphData || !Array.isArray(graphData)) {
      return { nodes: [], edges: [] };
    }
    
    // Extract nodes and edges
    const nodes = [];
    const edges = [];
    
    // Process nodes and edges based on API response format
    // This implementation will depend on the specific format of your graph data
    
    return { nodes, edges };
  };
  
  /**
   * Calculate graph metrics
   * @param {Array} graphData - Raw graph data from the API
   * @returns {Object} Graph metrics such as node count, edge count, etc.
   */
  export const calculateGraphMetrics = (graphData) => {
    if (!graphData || !Array.isArray(graphData)) {
      return { nodeCount: 0, edgeCount: 0, avgConnections: 0 };
    }
    
    // Calculate basic metrics
    const nodeCount = graphData.length;
    const edges = graphData.reduce((acc, node) => {
      return acc + (node.connections?.length || 0);
    }, 0);
    
    const avgConnections = nodeCount > 0 ? edges / nodeCount : 0;
    
    return {
      nodeCount,
      edgeCount: edges,
      avgConnections: avgConnections.toFixed(2)
    };
  };
  
  /**
   * Get color for graph node based on type or status
   * @param {string} nodeType - Type of node
   * @param {boolean} isActive - Whether the node is active
   * @returns {string} HEX color code
   */
  export const getNodeColor = (nodeType, isActive) => {
    if (isActive) {
      return '#4a90e2'; // Blue for active nodes
    }
    
    switch (nodeType) {
      case 'decision':
        return '#f44336'; // Red
      case 'action':
        return '#4caf50'; // Green
      case 'observation':
        return '#ff9800'; // Orange
      default:
        return '#6c757d'; // Gray for default
    }
  };
  
  /**
   * Format graph data for download
   * @param {Array} graphData - Raw graph data from the API
   * @param {string} format - Format type ('json', 'csv', 'graphml')
   * @returns {string} Formatted data string
   */
  export const formatGraphForDownload = (graphData, format = 'json') => {
    if (!graphData) {
      return '';
    }
    
    switch (format.toLowerCase()) {
      case 'json':
        return JSON.stringify(graphData, null, 2);
      case 'csv':
        // Simple CSV conversion example
        let csv = 'id,type,connections\n';
        graphData.forEach(node => {
          csv += `${node.id},${node.type},${(node.connections || []).join('|')}\n`;
        });
        return csv;
      case 'graphml':
        // GraphML format would require more complex processing
        return '<graphml>...</graphml>';
      default:
        return JSON.stringify(graphData);
    }
  };
  
  /**
   * Check if graph data has changed
   * @param {Array} oldData - Previous graph data
   * @param {Array} newData - New graph data
   * @returns {boolean} True if data has changed
   */
  export const hasGraphChanged = (oldData, newData) => {
    if (!oldData || !newData) return true;
    
    return JSON.stringify(oldData) !== JSON.stringify(newData);
  };
  
  export default {
    processGraphData,
    calculateGraphMetrics,
    getNodeColor,
    formatGraphForDownload,
    hasGraphChanged
  };