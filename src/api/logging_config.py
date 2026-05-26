import sys
import os
import json
import logging
from loguru import logger

def serialize_json(record):
    # Formats the Loguru log record into a structured JSON string
    log_data = {
        "timestamp": record["date"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "exception": None,
        **{k: v for k, v in record["extra"].items() if k != "serialized"}
    }
    if record["exception"]:
        log_data["exception"] = {
            "type": str(record["exception"].type),
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback
        }
    return json.dumps(log_data)

def formatter(record):
    # Determines formatting style.
    # If serialize is true, we pass serialize=True which logs JSON strings.
    if os.getenv("LOG_FORMAT", "TEXT").upper() == "JSON":
        record["extra"]["serialized"] = serialize_json(record)
        return "{extra[serialized]}\n"
    # Otherwise, clean human-readable output
    return "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"

def setup_logging():
    # Remove standard loguru handler
    logger.remove()
    
    # Configure Loguru
    logger.add(
        sys.stdout,
        format=formatter,
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        backtrace=True,
        diagnose=True
    )
    
    # Intercept standard library logging (e.g. Uvicorn logs)
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
                
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Direct uvicorn logs into loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logger_name = logging.getLogger(name)
        logger_name.handlers = [InterceptHandler()]
        logger_name.propagate = False
