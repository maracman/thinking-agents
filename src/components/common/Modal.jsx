import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import Button from './Button';

/**
 * Reusable Modal component
 * 
 * @param {Object} props
 * @param {boolean} props.isOpen - Whether the modal is open
 * @param {Function} props.onClose - Close handler
 * @param {string} [props.title] - Modal title
 * @param {React.ReactNode} props.children - Modal content
 * @param {string} [props.size='medium'] - Modal size (small, medium, large)
 * @param {boolean} [props.closeOnOutsideClick=true] - Whether to close on outside click
 * @param {boolean} [props.showCloseButton=true] - Whether to show the close button
 * @param {React.ReactNode} [props.footer] - Modal footer content
 * @param {string} [props.className] - Additional CSS classes
 */
const Modal = ({
  isOpen,
  onClose,
  title,
  children,
  size = 'medium',
  closeOnOutsideClick = true,
  showCloseButton = true,
  footer,
  className = '',
}) => {
  const modalRef = useRef(null);
  
  // Handle Escape key press
  useEffect(() => {
    const handleEscapeKey = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    
    document.addEventListener('keydown', handleEscapeKey);
    
    return () => {
      document.removeEventListener('keydown', handleEscapeKey);
    };
  }, [isOpen, onClose]);
  
  // Lock body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);
  
  // Close on outside click
  const handleBackdropClick = (e) => {
    if (closeOnOutsideClick && modalRef.current && !modalRef.current.contains(e.target)) {
      onClose();
    }
  };
  
  if (!isOpen) {
    return null;
  }
  
  // Construct CSS classes
  const modalClasses = [
    'modal',
    `modal-${size}`,
    className
  ].filter(Boolean).join(' ');
  
  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div 
        className={modalClasses} 
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? "modal-title" : undefined}
      >
        {(title || showCloseButton) && (
          <div className="modal-header">
            {title && (
              <h2 className="modal-title" id="modal-title">{title}</h2>
            )}
            
            {showCloseButton && (
              <Button 
                variant="icon" 
                onClick={onClose}
                aria-label="Close modal"
                className="modal-close-button"
              >
                <X size={20} />
              </Button>
            )}
          </div>
        )}
        
        <div className="modal-content">
          {children}
        </div>
        
        {footer && (
          <div className="modal-footer">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Modal.Footer - Specialized component for modal footers with buttons
 */
Modal.Footer = ({ children, className = '' }) => {
  const footerClasses = ['modal-footer', className].filter(Boolean).join(' ');
  return <div className={footerClasses}>{children}</div>;
};

/**
 * Modal.Actions - Container for modal action buttons
 */
Modal.Actions = ({ children, className = '' }) => {
  const actionsClasses = ['modal-actions', className].filter(Boolean).join(' ');
  return <div className={actionsClasses}>{children}</div>;
};

export default Modal;