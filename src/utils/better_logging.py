import logging
import sys
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'GRAY': '\033[90m',
        'CYAN': '\033[96m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'MAGENTA': '\033[95m',
        'WHITE': '\033[97m',
    }
    
    # Color mapping for each log level
    LEVEL_COLORS = {
        'DEBUG': COLORS['CYAN'],
        'INFO': COLORS['GREEN'],
        'WARNING': COLORS['YELLOW'],
        'ERROR': COLORS['RED'],
        'CRITICAL': COLORS['MAGENTA'],
    }
    
    # Color for metadata (timestamp, location)
    METADATA_COLOR = COLORS['GRAY']
    MESSAGE_COLOR = COLORS['WHITE']
    
    def format(self, record):
        # Get the log level color
        level_color = self.LEVEL_COLORS.get(record.levelname, self.COLORS['WHITE'])
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Format location (filename:line)
        location = f"{record.filename}:{record.lineno}"
        
        # Build the colored log message
        log_message = (
            f"{self.METADATA_COLOR}[{timestamp}]{self.COLORS['RESET']}"
            f"{self.METADATA_COLOR}({location}){self.COLORS['RESET']}"
            f"{level_color}[{record.levelname}]{self.COLORS['RESET']} "
            f"{self.MESSAGE_COLOR}{record.getMessage()}{self.COLORS['RESET']}"
        )
        
        # Add exception info if present
        if record.exc_info:
            log_message += f"\n{self.formatException(record.exc_info)}"
        
        return log_message


def get_logger(name=None, level=logging.INFO):
    """
    Get a configured logger that outputs to terminal with colored formatting.
    
    Args:
        name: Logger name (defaults to root logger if None)
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't already have handlers
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Set the colored formatter
        formatter = ColoredFormatter()
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
    
    return logger


if __name__ == "__main__":
    # Test the logger with all levels
    logger = get_logger(__name__, level=logging.DEBUG)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test with exception
    try:
        raise ValueError("Example exception for testing")
    except Exception as e:
        logger.exception("An exception occurred:")
