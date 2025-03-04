/**
 * Utility functions used throughout the application
 */

/**
 * Format a date to a readable string
 * @param {Date|string|number} date - Date to format
 * @param {Object} options - Formatting options
 * @returns {string} Formatted date string
 */
export const formatDate = (date, options = {}) => {
    if (!date) return '';
    
    const dateObj = typeof date === 'string' || typeof date === 'number' 
      ? new Date(date)
      : date;
    
    const defaultOptions = {
      includeTime: true,
      useRelative: false,
      shortFormat: false
    };
    
    const config = { ...defaultOptions, ...options };
    
    if (config.useRelative) {
      const now = new Date();
      const diffMs = now - dateObj;
      const diffSec = Math.round(diffMs / 1000);
      const diffMin = Math.round(diffSec / 60);
      const diffHours = Math.round(diffMin / 60);
      const diffDays = Math.round(diffHours / 24);
      
      if (diffSec < 60) return 'just now';
      if (diffMin < 60) return `${diffMin} minute${diffMin !== 1 ? 's' : ''} ago`;
      if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
      if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
    }
    
    if (config.shortFormat) {
      return dateObj.toLocaleDateString();
    }
    
    if (config.includeTime) {
      return dateObj.toLocaleString();
    }
    
    return dateObj.toLocaleDateString();
  };
  
  /**
   * Truncate text to a specified length
   * @param {string} text - Text to truncate
   * @param {number} maxLength - Maximum length
   * @param {string} suffix - Suffix to add when truncated (default: '...')
   * @returns {string} Truncated text
   */
  export const truncateText = (text, maxLength = 100, suffix = '...') => {
    if (!text || text.length <= maxLength) return text;
    
    return text.substring(0, maxLength).trim() + suffix;
  };
  
  /**
   * Generate a random ID
   * @param {number} length - ID length
   * @returns {string} Random ID
   */
  export const generateId = (length = 8) => {
    return Math.random().toString(36).substring(2, 2 + length);
  };
  
  /**
   * Capitalize first letter of a string
   * @param {string} str - String to capitalize
   * @returns {string} Capitalized string
   */
  export const capitalizeFirstLetter = (str) => {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
  };
  
  /**
   * Debounce a function call
   * @param {Function} func - Function to debounce
   * @param {number} wait - Wait time in milliseconds
   * @returns {Function} Debounced function
   */
  export const debounce = (func, wait = 300) => {
    let timeout;
    
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  };
  
  /**
   * Parse parameters from a URL query string
   * @param {string} queryString - Query string to parse
   * @returns {Object} Parsed parameters
   */
  export const parseQueryParams = (queryString = window.location.search) => {
    const params = {};
    const searchParams = new URLSearchParams(queryString);
    
    for (const [key, value] of searchParams.entries()) {
      params[key] = value;
    }
    
    return params;
  };
  
  /**
   * Check if value is empty (null, undefined, empty string, empty array, empty object)
   * @param {*} value - Value to check
   * @returns {boolean} Whether the value is empty
   */
  export const isEmpty = (value) => {
    return (
      value === null ||
      value === undefined ||
      value === '' ||
      (Array.isArray(value) && value.length === 0) ||
      (typeof value === 'object' && Object.keys(value).length === 0)
    );
  };
  
  /**
   * Format bytes to a human-readable string
   * @param {number} bytes - Bytes to format
   * @param {number} decimals - Decimal places
   * @returns {string} Formatted string
   */
  export const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };
  
  /**
   * Deep clone an object
   * @param {Object} obj - Object to clone
   * @returns {Object} Cloned object
   */
  export const deepClone = (obj) => {
    if (obj === null || typeof obj !== 'object') return obj;
    return JSON.parse(JSON.stringify(obj));
  };
  
  export default {
    formatDate,
    truncateText,
    generateId,
    capitalizeFirstLetter,
    debounce,
    parseQueryParams,
    isEmpty,
    formatBytes,
    deepClone
  };