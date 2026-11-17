"""Embedded SQLite persistence layer for OrderOps Agent."""

from orderops.db.database import DatabaseManager
from orderops.db.schema import SCHEMA_STATEMENTS

__all__ = ["DatabaseManager", "SCHEMA_STATEMENTS"]
