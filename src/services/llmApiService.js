/**
 * Service for interacting with the backend LLM API
 */

import axios from 'axios';

/**
 * Fetch the list of available LLM providers
 * @returns {Promise<Array>} Array of provider objects
 */
export const getLLMProviders = async () => {
  try {
    const response = await axios.get('/get_llm_providers');
    return response.data.providers || [];
  } catch (error) {
    console.error('Error fetching LLM providers:', error);
    throw error;
  }
};

/**
 * Fetch the list of available models for a provider
 * @param {string} provider - Provider ID
 * @returns {Promise<Array>} Array of model objects
 */
export const getLLMModels = async (provider) => {
  try {
    const response = await axios.get(`/get_llm_models?provider=${provider}`);
    return response.data.models || [];
  } catch (error) {
    console.error(`Error fetching models for ${provider}:`, error);
    throw error;
  }
};

/**
 * Get current LLM settings
 * @returns {Promise<Object>} Current LLM settings
 */
export const getLLMSettings = async () => {
  try {
    const response = await axios.get('/get_llm_settings');
    return response.data.settings || {};
  } catch (error) {
    console.error('Error fetching LLM settings:', error);
    throw error;
  }
};

/**
 * Update LLM settings
 * @param {Object} settings - New settings
 * @returns {Promise<Object>} Updated settings
 */
export const updateLLMSettings = async (settings) => {
  try {
    const response = await axios.post('/update_llm_settings', settings);
    return response.data;
  } catch (error) {
    console.error('Error updating LLM settings:', error);
    throw error;
  }
};

/**
 * Test LLM configuration
 * @param {Object} config - LLM configuration
 * @returns {Promise<Object>} Test results
 */
export const testLLMConfiguration = async (config) => {
  try {
    const response = await axios.post('/test_llm_configuration', config);
    return response.data;
  } catch (error) {
    console.error('Error testing LLM configuration:', error);
    
    // Extract error message from response if available
    const errorMessage = error.response?.data?.error || 'Unknown error occurred';
    throw new Error(errorMessage);
  }
};

export default {
  getLLMProviders,
  getLLMModels,
  getLLMSettings,
  updateLLMSettings,
  testLLMConfiguration
};