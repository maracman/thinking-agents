import React from 'react';

/**
 * Reusable Button component with multiple variants
 * 
 * @param {Object} props
 * @param {string} [props.variant='primary'] - Button variant (primary, secondary, danger, success)
 * @param {string} [props.size='medium'] - Button size (small, medium, large)
 * @param {boolean} [props.disabled=false] - Whether the button is disabled
 * @param {boolean} [props.fullWidth=false] - Whether the button should take full width
 * @param {Function} [props.onClick] - Click handler
 * @param {React.ReactNode} [props.icon] - Optional icon to display
 * @param {string} [props.iconPosition='left'] - Icon position (left or right)
 * @param {string} [props.type='button'] - Button type attribute
 * @param {string} [props.className] - Additional CSS classes
 * @param {React.ReactNode} props.children - Button content
 */
const Button = ({
  variant = 'primary',
  size = 'medium',
  disabled = false,
  fullWidth = false,
  onClick,
  icon,
  iconPosition = 'left',
  type = 'button',
  className = '',
  children,
  ...rest
}) => {
  // Construct CSS classes based on props
  const buttonClasses = [
    'dynamic-button',
    variant,
    size,
    fullWidth ? 'full-width' : '',
    className
  ].filter(Boolean).join(' ');
  
  return (
    <button
      type={type}
      className={buttonClasses}
      onClick={onClick}
      disabled={disabled}
      {...rest}
    >
      {icon && iconPosition === 'left' && (
        <span className="button-icon left">{icon}</span>
      )}
      
      {children && (
        <span className="button-text">{children}</span>
      )}
      
      {icon && iconPosition === 'right' && (
        <span className="button-icon right">{icon}</span>
      )}
    </button>
  );
};

export default Button;