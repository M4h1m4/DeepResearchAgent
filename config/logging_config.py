import logging 
import sys 

from pythonjsonlogger import jsonlogger 

JSONFormatter = jsonlogger.JsonFormatter

def setup_json_logging(log_level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.handlers = []
    handler = logging.StreamHandler(sys.stdout)

    formatter = JSONFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Map environment names to logging levels
    level_mapping = {
        "DEVELOPMENT": "DEBUG",
        "PRODUCTION": "INFO",
        "TEST": "DEBUG",
        "STAGING": "INFO"
    }
    
    # Normalize log level
    normalized_level = log_level.upper()
    if normalized_level in level_mapping:
        normalized_level = level_mapping[normalized_level]
    
    # Validate and set level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if normalized_level not in valid_levels:
        normalized_level = "INFO"  # Default fallback
    
    root_logger.setLevel(getattr(logging, normalized_level))

    root_logger.propagate = False 

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name) #get a logger with the given name and then return the instance