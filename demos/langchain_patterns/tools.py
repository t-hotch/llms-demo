"""Tools for ReAct agent demo.

This module provides simple tools for the ReAct agent to demonstrate
multi-step reasoning with tool use:
- Calculator for arithmetic operations
- Date utilities for time-based calculations
"""

from datetime import datetime, timedelta
from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """Performs arithmetic calculations on mathematical expressions.
    
    Use this tool when you need to calculate numbers, percentages, or solve
    math problems. Supports addition (+), subtraction (-), multiplication (*),
    division (/), modulo (%), and exponentiation (**).
    
    Args:
        expression: A mathematical expression as a string (e.g., "15 * 0.15", "100 + 50")
    
    Returns:
        The calculated result as a string
    
    Examples:
        calculator("25 * 0.15") -> "3.75"
        calculator("(100 + 50) / 2") -> "75.0"
        calculator("2 ** 8") -> "256"
    """
    try:
        # Evaluate the mathematical expression
        result = eval(expression)
        return str(result)
    
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def get_current_date() -> str:
    """Returns today's date in YYYY-MM-DD format.
    
    Use this tool when you need to know what today's date is, especially for
    calculating time differences or determining days until a future date.
    
    Returns:
        Today's date as a string in ISO format (YYYY-MM-DD)
    
    Example:
        get_current_date() -> "2026-03-16"
    """
    return datetime.now().strftime("%Y-%m-%d")


@tool
def days_between(start_date: str, end_date: str) -> str:
    """Calculates the number of days between two dates.
    
    Use this tool to find out how many days are between two specific dates.
    Useful for calculating age in days, countdown to events, or time differences.
    
    Args:
        start_date: Starting date in YYYY-MM-DD format (e.g., "2026-03-16")
        end_date: Ending date in YYYY-MM-DD format (e.g., "2026-12-25")
    
    Returns:
        The number of days between the dates as a string (positive if end_date is after start_date)
    
    Examples:
        days_between("2026-03-16", "2026-12-25") -> "284"
        days_between("1990-03-15", "2026-03-16") -> "13150"
    """
    try:
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate difference
        diff = (end - start).days
        
        return str(diff)
    
    except ValueError as e:
        return f"Error parsing dates: {str(e)}. Please use YYYY-MM-DD format."


# List of all tools for easy import
TOOLS = [calculator, get_current_date, days_between]
