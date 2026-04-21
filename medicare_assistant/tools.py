"""
Tools for MediCare Hospital Assistant.
Rules:
  - Tools NEVER raise exceptions — always return a string (including errors).
  - Tools handle what the KB cannot: live datetime, arithmetic, web lookup.
"""

from datetime import datetime


def get_current_datetime() -> str:
    """Return current date and time as a readable string."""
    try:
        now = datetime.now()
        return (
            f"Current date: {now.strftime('%A, %d %B %Y')}\n"
            f"Current time: {now.strftime('%I:%M %p')}"
        )
    except Exception as e:
        return f"Error retrieving date/time: {str(e)}"


def calculate(expression: str) -> str:
    """
    Safely evaluate a basic arithmetic expression.
    Accepts expressions like '300 + 600', '2499 * 2', etc.
    """
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: only basic arithmetic is supported (+, -, *, /, parentheses)."
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return f"Result of '{expression}' = {result}"
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Error in calculation: {str(e)}"


def get_emergency_info() -> str:
    """Return emergency contact information immediately."""
    return (
        "EMERGENCY CONTACTS — MediCare General Hospital:\n"
        "Emergency Department (24/7): 040-99999999\n"
        "Ambulance: 040-99999999\n"
        "Main Helpline: 040-12345678\n"
        "The Emergency Department is open 24 hours, 7 days a week."
    )


# Tool registry — maps tool name to callable
TOOLS = {
    "datetime": get_current_datetime,
    "calculator": calculate,
    "emergency": get_emergency_info,
}


def run_tool(tool_name: str, tool_input: str = "") -> str:
    """Dispatch to the correct tool. Always returns a string."""
    if tool_name not in TOOLS:
        return f"Unknown tool '{tool_name}'. Available tools: {list(TOOLS.keys())}"
    try:
        fn = TOOLS[tool_name]
        if tool_name == "calculator":
            return fn(tool_input)
        return fn()
    except Exception as e:
        return f"Tool '{tool_name}' encountered an error: {str(e)}"
