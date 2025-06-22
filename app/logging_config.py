import logging
import re
import sys

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    def format(self, record):
        if record.levelno == logging.DEBUG and isinstance(record.msg, str) and "MODEL MAPPING" in record.msg:
            return f"{Colors.BOLD}{Colors.GREEN}{record.msg}{Colors.RESET}"
        return super().format(record)

# Filter to block noisy log messages
class MessageFilter(logging.Filter):
    def filter(self, record):
        blocked_phrases = [
            "LiteLLM completion()", "HTTP Request:",
            "selected model name for cost calculation", "utils.py", "cost_calculator"
        ]
        message = record.getMessage()
        return not any(phrase in message for phrase in blocked_phrases)

# Filter to redact sensitive tokens from log messages
class RedactAPIKeyFilter(logging.Filter):
    _patterns = [
        (re.compile(r"key=[A-Za-z0-9_\-]+"), "key=***"),
        (re.compile(r"Bearer [A-Za-z0-9_\-]+"), "Bearer ***"),
        (re.compile(r"sk-[A-Za-z0-9]{20,}"), "sk-***"),
        (re.compile(r"AIza[A-Za-z0-9_-]+"), "AIza***"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "msg") and isinstance(record.msg, str):
            msg = record.msg
            for pat, repl in self._patterns:
                msg = pat.sub(repl, msg)
            record.msg = msg
        return True

def setup_logging():
    """Configures the root logger and other loggers for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    # Apply filters to the root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(MessageFilter())
    root_logger.addFilter(RedactAPIKeyFilter())

    # Apply custom formatter to the console handler
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Silence noisy third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

def log_request_beautifully(method, path, original_model_display, mapped_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing model mapping."""
    original_display = f"{Colors.CYAN}{original_model_display}{Colors.RESET}"
    endpoint = path.split("?")[0]

    target_provider, target_model_name = "unknown", mapped_model
    target_color = Colors.GREEN  # Default

    if "/" in mapped_model:
        try:
            provider, model_name = mapped_model.split("/", 1)
            target_provider, target_model_name = provider, model_name
            color_map = {
                "openai": Colors.GREEN, "openrouter": Colors.GREEN,
                "gemini": Colors.YELLOW, "vertex_ai": Colors.BLUE,
                "xai": Colors.CYAN, "anthropic": Colors.RED
            }
            target_color = color_map.get(provider, Colors.GREEN)
        except ValueError:
            logging.getLogger(__name__).warning(f"Could not parse provider from mapped model: {mapped_model}")

    target_display = f"{target_color}{target_model_name}{Colors.RESET}"
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"

    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{original_display} → {target_display} ({target_provider}) {tools_str} {messages_str}"

    print(log_line)
    print(model_line)
    sys.stdout.flush()
