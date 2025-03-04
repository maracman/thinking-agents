import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';

/**
 * Reusable Dropdown component
 * 
 * @param {Object} props
 * @param {string} [props.label] - Label for the dropdown
 * @param {Array} props.options - Array of option objects { value, label, icon? }
 * @param {*} [props.value] - Currently selected value
 * @param {Function} props.onChange - Change handler
 * @param {boolean} [props.disabled=false] - Whether the dropdown is disabled
 * @param {string} [props.placeholder='Select...'] - Placeholder text
 * @param {boolean} [props.searchable=false] - Whether dropdown is searchable
 * @param {string} [props.className] - Additional CSS classes
 * @param {boolean} [props.fullWidth=false] - Whether the dropdown should take full width
 */
const Dropdown = ({
  label,
  options = [],
  value,
  onChange,
  disabled = false,
  placeholder = 'Select...',
  searchable = false,
  className = '',
  fullWidth = false,
  ...rest
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const dropdownRef = useRef(null);
  
  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  // Find currently selected option
  const selectedOption = options.find(option => option.value === value);
  
  // Toggle dropdown
  const handleToggle = () => {
    if (!disabled) {
      setIsOpen(!isOpen);
      setSearchTerm('');
    }
  };
  
  // Handle option selection
  const handleSelect = (option) => {
    onChange(option.value);
    setIsOpen(false);
    setSearchTerm('');
  };
  
  // Handle search input
  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
  };
  
  // Filter options based on search term
  const filteredOptions = searchTerm
    ? options.filter(option => 
        option.label.toLowerCase().includes(searchTerm.toLowerCase()))
    : options;
  
  // Construct CSS classes
  const dropdownClasses = [
    'dropdown',
    fullWidth ? 'full-width' : '',
    disabled ? 'disabled' : '',
    isOpen ? 'open' : '',
    className
  ].filter(Boolean).join(' ');
  
  return (
    <div className={dropdownClasses} ref={dropdownRef} {...rest}>
      {label && <label className="dropdown-label">{label}</label>}
      
      <div 
        className="dropdown-toggle" 
        onClick={handleToggle}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span className="dropdown-value">
          {selectedOption ? (
            <>
              {selectedOption.icon && (
                <span className="dropdown-option-icon">{selectedOption.icon}</span>
              )}
              <span>{selectedOption.label}</span>
            </>
          ) : (
            <span className="dropdown-placeholder">{placeholder}</span>
          )}
        </span>
        <ChevronDown size={18} className={`dropdown-arrow ${isOpen ? 'open' : ''}`} />
      </div>
      
      {isOpen && (
        <div className="dropdown-content">
          {searchable && (
            <div className="dropdown-search">
              <input
                type="text"
                placeholder="Search..."
                value={searchTerm}
                onChange={handleSearch}
                onClick={(e) => e.stopPropagation()}
                autoFocus
              />
            </div>
          )}
          
          <ul className="dropdown-menu" role="listbox">
            {filteredOptions.map((option) => (
              <li
                key={option.value}
                className={`dropdown-option ${option.value === value ? 'selected' : ''}`}
                onClick={() => handleSelect(option)}
                role="option"
                aria-selected={option.value === value}
              >
                {option.icon && (
                  <span className="dropdown-option-icon">{option.icon}</span>
                )}
                <span>{option.label}</span>
              </li>
            ))}
            
            {filteredOptions.length === 0 && (
              <li className="dropdown-no-results">No results found</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Dropdown;