"""
Watchlist Manager with SQLite Persistence

Provides CRUD operations for watchlists with local SQLite storage.
Supports notes, price targets, alerts, and import/export functionality.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WatchlistItem:
    """Represents a single item in a watchlist."""
    id: Optional[int] = None
    ticker: str = ""
    company_name: str = ""
    notes: str = ""
    price_target: Optional[float] = None
    price_target_type: str = "above"  # 'above', 'below', 'between'
    price_target_high: Optional[float] = None
    alert_enabled: bool = False
    alert_score_threshold: Optional[float] = None
    alert_channels: str = "dashboard"  # comma-separated: 'dashboard,email,slack,sms'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: str = ""  # comma-separated tags

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "notes": self.notes,
            "price_target": self.price_target,
            "price_target_type": self.price_target_type,
            "price_target_high": self.price_target_high,
            "alert_enabled": self.alert_enabled,
            "alert_score_threshold": self.alert_score_threshold,
            "alert_channels": self.alert_channels,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatchlistItem":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=data.get("id"),
            ticker=data.get("ticker", ""),
            company_name=data.get("company_name", ""),
            notes=data.get("notes", ""),
            price_target=data.get("price_target"),
            price_target_type=data.get("price_target_type", "above"),
            price_target_high=data.get("price_target_high"),
            alert_enabled=data.get("alert_enabled", False),
            alert_score_threshold=data.get("alert_score_threshold"),
            alert_channels=data.get("alert_channels", "dashboard"),
            created_at=created_at,
            updated_at=updated_at,
            tags=data.get("tags", ""),
        )

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "WatchlistItem":
        """Create from database row."""
        return cls(
            id=row["id"],
            ticker=row["ticker"],
            company_name=row["company_name"] or "",
            notes=row["notes"] or "",
            price_target=row["price_target"],
            price_target_type=row["price_target_type"] or "above",
            price_target_high=row["price_target_high"],
            alert_enabled=bool(row["alert_enabled"]),
            alert_score_threshold=row["alert_score_threshold"],
            alert_channels=row["alert_channels"] or "dashboard",
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            tags=row["tags"] or "",
        )

    def get_tags_list(self) -> List[str]:
        """Get tags as a list."""
        if not self.tags:
            return []
        return [t.strip() for t in self.tags.split(",") if t.strip()]

    def set_tags_list(self, tags: List[str]) -> None:
        """Set tags from a list."""
        self.tags = ",".join(t.strip() for t in tags if t.strip())

    def get_alert_channels_list(self) -> List[str]:
        """Get alert channels as a list."""
        if not self.alert_channels:
            return ["dashboard"]
        return [c.strip() for c in self.alert_channels.split(",") if c.strip()]

    def set_alert_channels_list(self, channels: List[str]) -> None:
        """Set alert channels from a list."""
        self.alert_channels = ",".join(c.strip() for c in channels if c.strip())


@dataclass
class Watchlist:
    """Represents a watchlist collection."""
    id: Optional[int] = None
    name: str = "Default"
    description: str = ""
    is_default: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    items: List[WatchlistItem] = field(default_factory=list)

    def to_dict(self, include_items: bool = True) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_items:
            data["items"] = [item.to_dict() for item in self.items]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Watchlist":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        items = [
            WatchlistItem.from_dict(item)
            for item in data.get("items", [])
        ]

        return cls(
            id=data.get("id"),
            name=data.get("name", "Default"),
            description=data.get("description", ""),
            is_default=data.get("is_default", False),
            created_at=created_at,
            updated_at=updated_at,
            items=items,
        )


class WatchlistManager:
    """
    Manages watchlists with SQLite persistence.

    Provides CRUD operations and import/export functionality.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize watchlist manager.

        Args:
            db_path: Path to SQLite database file. Defaults to user data directory.
        """
        if db_path is None:
            # Default to user data directory
            data_dir = Path.home() / ".investment_dashboard"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "watchlists.db")

        self.db_path = db_path
        self._initialize_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create watchlists table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    is_default INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # Create watchlist_items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    notes TEXT,
                    price_target REAL,
                    price_target_type TEXT DEFAULT 'above',
                    price_target_high REAL,
                    alert_enabled INTEGER DEFAULT 0,
                    alert_score_threshold REAL,
                    alert_channels TEXT DEFAULT 'dashboard',
                    created_at TEXT,
                    updated_at TEXT,
                    tags TEXT,
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE,
                    UNIQUE(watchlist_id, ticker)
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_ticker ON watchlist_items(ticker)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_watchlist ON watchlist_items(watchlist_id)
            """)

            # Create default watchlist if none exists
            cursor.execute("SELECT COUNT(*) FROM watchlists")
            if cursor.fetchone()[0] == 0:
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO watchlists (name, description, is_default, created_at, updated_at)
                    VALUES (?, ?, 1, ?, ?)
                """, ("Default", "Default watchlist", now, now))

            conn.commit()

    # Watchlist CRUD operations

    def create_watchlist(self, name: str, description: str = "") -> Watchlist:
        """
        Create a new watchlist.

        Args:
            name: Watchlist name
            description: Watchlist description

        Returns:
            Created watchlist
        """
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO watchlists (name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (name, description, now, now))

            return Watchlist(
                id=cursor.lastrowid,
                name=name,
                description=description,
                created_at=datetime.fromisoformat(now),
                updated_at=datetime.fromisoformat(now),
            )

    def get_watchlist(self, watchlist_id: int) -> Optional[Watchlist]:
        """
        Get a watchlist by ID.

        Args:
            watchlist_id: Watchlist ID

        Returns:
            Watchlist or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM watchlists WHERE id = ?", (watchlist_id,))
            row = cursor.fetchone()

            if not row:
                return None

            watchlist = Watchlist(
                id=row["id"],
                name=row["name"],
                description=row["description"] or "",
                is_default=bool(row["is_default"]),
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            )

            # Get items
            cursor.execute("""
                SELECT * FROM watchlist_items
                WHERE watchlist_id = ?
                ORDER BY created_at DESC
            """, (watchlist_id,))

            watchlist.items = [
                WatchlistItem.from_db_row(item_row)
                for item_row in cursor.fetchall()
            ]

            return watchlist

    def get_default_watchlist(self) -> Watchlist:
        """
        Get the default watchlist.

        Returns:
            Default watchlist (creates one if none exists)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM watchlists WHERE is_default = 1 LIMIT 1")
            row = cursor.fetchone()

            if row:
                return self.get_watchlist(row["id"])

            # Create default if doesn't exist
            return self.create_watchlist("Default", "Default watchlist")

    def get_all_watchlists(self) -> List[Watchlist]:
        """
        Get all watchlists.

        Returns:
            List of watchlists (without items for performance)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM watchlists ORDER BY is_default DESC, name")

            watchlists = []
            for row in cursor.fetchall():
                watchlist = Watchlist(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"] or "",
                    is_default=bool(row["is_default"]),
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                    updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
                )
                watchlists.append(watchlist)

            return watchlists

    def update_watchlist(
        self,
        watchlist_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """
        Update a watchlist.

        Args:
            watchlist_id: Watchlist ID
            name: New name
            description: New description

        Returns:
            True if updated
        """
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(watchlist_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE watchlists SET {', '.join(updates)} WHERE id = ?",
                params
            )
            return cursor.rowcount > 0

    def delete_watchlist(self, watchlist_id: int) -> bool:
        """
        Delete a watchlist.

        Args:
            watchlist_id: Watchlist ID

        Returns:
            True if deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Don't delete if it's the only watchlist
            cursor.execute("SELECT COUNT(*) FROM watchlists")
            if cursor.fetchone()[0] <= 1:
                logger.warning("Cannot delete the only watchlist")
                return False

            cursor.execute("DELETE FROM watchlists WHERE id = ? AND is_default = 0", (watchlist_id,))
            return cursor.rowcount > 0

    # Watchlist Item CRUD operations

    def add_item(
        self,
        watchlist_id: int,
        ticker: str,
        company_name: str = "",
        notes: str = "",
        price_target: Optional[float] = None,
        alert_enabled: bool = False,
        alert_score_threshold: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[WatchlistItem]:
        """
        Add an item to a watchlist.

        Args:
            watchlist_id: Watchlist ID
            ticker: Stock ticker
            company_name: Company name
            notes: Notes
            price_target: Price target
            alert_enabled: Enable alerts
            alert_score_threshold: Score threshold for alerts
            tags: Item tags

        Returns:
            Created item or None if already exists
        """
        ticker = ticker.upper().strip()
        now = datetime.now().isoformat()
        tags_str = ",".join(tags) if tags else ""

        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    INSERT INTO watchlist_items (
                        watchlist_id, ticker, company_name, notes,
                        price_target, alert_enabled, alert_score_threshold,
                        tags, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    watchlist_id, ticker, company_name, notes,
                    price_target, int(alert_enabled), alert_score_threshold,
                    tags_str, now, now
                ))

                return WatchlistItem(
                    id=cursor.lastrowid,
                    ticker=ticker,
                    company_name=company_name,
                    notes=notes,
                    price_target=price_target,
                    alert_enabled=alert_enabled,
                    alert_score_threshold=alert_score_threshold,
                    tags=tags_str,
                    created_at=datetime.fromisoformat(now),
                    updated_at=datetime.fromisoformat(now),
                )
            except sqlite3.IntegrityError:
                logger.warning(f"Item {ticker} already exists in watchlist {watchlist_id}")
                return None

    def get_item(self, item_id: int) -> Optional[WatchlistItem]:
        """
        Get a watchlist item by ID.

        Args:
            item_id: Item ID

        Returns:
            Item or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM watchlist_items WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            return WatchlistItem.from_db_row(row) if row else None

    def get_item_by_ticker(
        self,
        watchlist_id: int,
        ticker: str,
    ) -> Optional[WatchlistItem]:
        """
        Get a watchlist item by ticker.

        Args:
            watchlist_id: Watchlist ID
            ticker: Stock ticker

        Returns:
            Item or None
        """
        ticker = ticker.upper().strip()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM watchlist_items
                WHERE watchlist_id = ? AND ticker = ?
            """, (watchlist_id, ticker))
            row = cursor.fetchone()
            return WatchlistItem.from_db_row(row) if row else None

    def update_item(
        self,
        item_id: int,
        notes: Optional[str] = None,
        price_target: Optional[float] = None,
        price_target_type: Optional[str] = None,
        price_target_high: Optional[float] = None,
        alert_enabled: Optional[bool] = None,
        alert_score_threshold: Optional[float] = None,
        alert_channels: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Update a watchlist item.

        Args:
            item_id: Item ID
            notes: New notes
            price_target: New price target
            price_target_type: New price target type
            price_target_high: New high price target
            alert_enabled: Enable/disable alerts
            alert_score_threshold: New score threshold
            alert_channels: New alert channels
            tags: New tags

        Returns:
            True if updated
        """
        updates = []
        params = []

        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)

        if price_target is not None:
            updates.append("price_target = ?")
            params.append(price_target)

        if price_target_type is not None:
            updates.append("price_target_type = ?")
            params.append(price_target_type)

        if price_target_high is not None:
            updates.append("price_target_high = ?")
            params.append(price_target_high)

        if alert_enabled is not None:
            updates.append("alert_enabled = ?")
            params.append(int(alert_enabled))

        if alert_score_threshold is not None:
            updates.append("alert_score_threshold = ?")
            params.append(alert_score_threshold)

        if alert_channels is not None:
            updates.append("alert_channels = ?")
            params.append(",".join(alert_channels))

        if tags is not None:
            updates.append("tags = ?")
            params.append(",".join(tags))

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(item_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE watchlist_items SET {', '.join(updates)} WHERE id = ?",
                params
            )
            return cursor.rowcount > 0

    def remove_item(self, item_id: int) -> bool:
        """
        Remove an item from a watchlist.

        Args:
            item_id: Item ID

        Returns:
            True if removed
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM watchlist_items WHERE id = ?", (item_id,))
            return cursor.rowcount > 0

    def remove_item_by_ticker(self, watchlist_id: int, ticker: str) -> bool:
        """
        Remove an item by ticker.

        Args:
            watchlist_id: Watchlist ID
            ticker: Stock ticker

        Returns:
            True if removed
        """
        ticker = ticker.upper().strip()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM watchlist_items
                WHERE watchlist_id = ? AND ticker = ?
            """, (watchlist_id, ticker))
            return cursor.rowcount > 0

    def is_in_watchlist(self, watchlist_id: int, ticker: str) -> bool:
        """
        Check if ticker is in watchlist.

        Args:
            watchlist_id: Watchlist ID
            ticker: Stock ticker

        Returns:
            True if in watchlist
        """
        ticker = ticker.upper().strip()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM watchlist_items
                WHERE watchlist_id = ? AND ticker = ?
            """, (watchlist_id, ticker))
            return cursor.fetchone() is not None

    def get_tickers(self, watchlist_id: int) -> List[str]:
        """
        Get all tickers in a watchlist.

        Args:
            watchlist_id: Watchlist ID

        Returns:
            List of tickers
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ticker FROM watchlist_items
                WHERE watchlist_id = ?
                ORDER BY ticker
            """, (watchlist_id,))
            return [row["ticker"] for row in cursor.fetchall()]

    def get_items_with_alerts(self) -> List[WatchlistItem]:
        """
        Get all items with alerts enabled.

        Returns:
            List of items with alerts enabled
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM watchlist_items
                WHERE alert_enabled = 1
            """)
            return [WatchlistItem.from_db_row(row) for row in cursor.fetchall()]

    # Import/Export

    def export_watchlist(self, watchlist_id: int) -> str:
        """
        Export a watchlist to JSON.

        Args:
            watchlist_id: Watchlist ID

        Returns:
            JSON string
        """
        watchlist = self.get_watchlist(watchlist_id)
        if not watchlist:
            return "{}"
        return json.dumps(watchlist.to_dict(), indent=2)

    def import_watchlist(self, json_data: str, merge: bool = False) -> Optional[Watchlist]:
        """
        Import a watchlist from JSON.

        Args:
            json_data: JSON string
            merge: Merge with existing watchlist of same name

        Returns:
            Imported watchlist or None on error
        """
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return None

        name = data.get("name", "Imported")
        description = data.get("description", "")
        items = data.get("items", [])

        # Check if watchlist with name exists
        existing = None
        for wl in self.get_all_watchlists():
            if wl.name == name:
                existing = wl
                break

        if existing and merge:
            watchlist_id = existing.id
        elif existing:
            # Create with unique name
            counter = 1
            new_name = f"{name} ({counter})"
            while any(wl.name == new_name for wl in self.get_all_watchlists()):
                counter += 1
                new_name = f"{name} ({counter})"
            watchlist = self.create_watchlist(new_name, description)
            watchlist_id = watchlist.id
        else:
            watchlist = self.create_watchlist(name, description)
            watchlist_id = watchlist.id

        # Add items
        for item_data in items:
            self.add_item(
                watchlist_id=watchlist_id,
                ticker=item_data.get("ticker", ""),
                company_name=item_data.get("company_name", ""),
                notes=item_data.get("notes", ""),
                price_target=item_data.get("price_target"),
                alert_enabled=item_data.get("alert_enabled", False),
                alert_score_threshold=item_data.get("alert_score_threshold"),
                tags=item_data.get("tags", "").split(",") if item_data.get("tags") else None,
            )

        return self.get_watchlist(watchlist_id)

    def export_all(self) -> str:
        """
        Export all watchlists to JSON.

        Returns:
            JSON string
        """
        watchlists = []
        for wl in self.get_all_watchlists():
            full_wl = self.get_watchlist(wl.id)
            if full_wl:
                watchlists.append(full_wl.to_dict())

        return json.dumps({
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "watchlists": watchlists,
        }, indent=2)


# Global instance for convenience
_manager: Optional[WatchlistManager] = None


def get_watchlist_manager(db_path: Optional[str] = None) -> WatchlistManager:
    """
    Get or create the global watchlist manager.

    Args:
        db_path: Database path (only used on first call)

    Returns:
        WatchlistManager instance
    """
    global _manager
    if _manager is None:
        _manager = WatchlistManager(db_path)
    return _manager
