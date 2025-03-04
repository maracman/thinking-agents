import React, { useState, useEffect } from 'react';

/**
 * Reusable Slider component
 * 
 * @param {Object} props
 * @param {string} [props.id] - Input ID
 * @param {string} [props.name] - Input name
 * @param {string} [props.label] - Label for the slider
 * @param {number} [props.min=0] - Minimum value
 * @param {number} [props.max=100] - Maximum value
 * @param {number} [props.step=1] - Step size
 * @param {number} props.value - Current value
 * @param {Function} props.onChange - Change handler
 * @param {boolean} [props.disabled=false] - Whether the slider is disabled
 * @param {boolean} [props.showValue=true] - Whether to show the current value
 * @param {string} [props.className] - Additional CSS classes
 * @param {string} [props.valuePrefix=''] - Prefix for displayed value
 * @param {string} [props.valueSuffix=''] - Suffix for displayed value
 */
const Slider = ({
  id,
  name,
  label,
  min = 0,
  max = 100,
  step = 1,
  value,
  onChange,
  disabled = false,
  showValue = true,
  className = '',
  valuePrefix = '',
  valueSuffix = '',
  ...rest
}) => {
  const [localValue, setLocalValue] = useState(value);
  
  // Update local value when prop changes
  useEffect(() => {
    setLocalValue(value);
  }, [value]);
  
  // Format display value
  const displayValue = `${valuePrefix}${Number(localValue).toFixed(
    Number.isInteger(step) ? 0 : String(step).split('.')[1]?.length || 0
  )}${valueSuffix}`;
  
  // Handle slider change
  const handleChange = (e) => {
    const newValue = parseFloat(e.target.value);
    setLocalValue(newValue);
    onChange(newValue);
  };
  
  // Construct CSS classes
  const sliderClasses = [
    'range-slider',
    disabled ? 'disabled' : '',
    className
  ].filter(Boolean).join(' ');
  
  // Calculate position percentage for value label (0-100%)
  const valuePosition = ((localValue - min) / (max - min)) * 100;
  
  return (
    <div className="input-group compact">
      {label && (
        <label 
          htmlFor={id} 
          className="slider-label"
        >
          {label}
        </label>
      )}
      
      <div className={sliderClasses}>
        <input
          type="range"
          id={id}
          name={name}
          className="range-slider__range"
          min={min}
          max={max}
          step={step}
          value={localValue}
          onChange={handleChange}
          disabled={disabled}
          {...rest}
        />
        
        {showValue && (
          <div 
            className="range-slider__value"
            style={{ 
              left: `${Math.min(Math.max(valuePosition, 10), 90)}%`,
              transform: 'translateX(-50%)'
            }}
          >
            {displayValue}
          </div>
        )}
      </div>
    </div>
  );
};

export default Slider;