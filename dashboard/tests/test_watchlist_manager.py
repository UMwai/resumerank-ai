"""
Tests for Watchlist Manager Module

Tests cover:
- WatchlistItem dataclass
- Watchlist dataclass
- WatchlistManager CRUD operations
- Import/Export functionality
"""

import json
import os
import pytest
import tempfile
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.watchlist_manager import (
    WatchlistItem,
    Watchlist,
    WatchlistManager,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def manager(temp_db):
    """Create a WatchlistManager with temporary database."""
    return WatchlistManager(db_path=temp_db)


class TestWatchlistItem:
    """Test WatchlistItem dataclass."""

    def test_item_creation(self):
        """Test creating a WatchlistItem."""
        item = WatchlistItem(
            ticker="MRNA",
            company_name="Moderna Inc.",
            notes="Monitoring for vaccine updates",
            price_target=150.0,
            alert_enabled=True,
        )

        assert item.ticker == "MRNA"
        assert item.company_name == "Moderna Inc."
        assert item.notes == "Monitoring for vaccine updates"
        assert item.price_target == 150.0
        assert item.alert_enabled is True

    def test_item_to_dict(self):
        """Test WatchlistItem to_dict conversion."""
        item = WatchlistItem(
            id=1,
            ticker="PFE",
            company_name="Pfizer",
            price_target=45.0,
            tags="pharma,vaccine",
        )

        data = item.to_dict()

        assert data["id"] == 1
        assert data["ticker"] == "PFE"
        assert data["company_name"] == "Pfizer"
        assert data["price_target"] == 45.0
        assert data["tags"] == "pharma,vaccine"

    def test_item_from_dict(self):
        """Test WatchlistItem from_dict creation."""
        data = {
            "ticker": "GILD",
            "company_name": "Gilead Sciences",
            "notes": "HIV pipeline strong",
            "price_target": 85.0,
            "alert_enabled": True,
            "tags": "biotech,hiv",
        }

        item = WatchlistItem.from_dict(data)

        assert item.ticker == "GILD"
        assert item.company_name == "Gilead Sciences"
        assert item.notes == "HIV pipeline strong"
        assert item.alert_enabled is True

    def test_get_tags_list(self):
        """Test getting tags as list."""
        item = WatchlistItem(tags="biotech,pharma,vaccine")
        tags = item.get_tags_list()

        assert tags == ["biotech", "pharma", "vaccine"]

    def test_get_tags_list_empty(self):
        """Test getting tags as list when empty."""
        item = WatchlistItem(tags="")
        tags = item.get_tags_list()

        assert tags == []

    def test_set_tags_list(self):
        """Test setting tags from list."""
        item = WatchlistItem()
        item.set_tags_list(["biotech", "covid", "mrna"])

        assert item.tags == "biotech,covid,mrna"


class TestWatchlist:
    """Test Watchlist dataclass."""

    def test_watchlist_creation(self):
        """Test creating a Watchlist."""
        watchlist = Watchlist(
            name="Biotech Portfolio",
            description="High-conviction biotech plays",
        )

        assert watchlist.name == "Biotech Portfolio"
        assert watchlist.description == "High-conviction biotech plays"
        assert watchlist.items == []

    def test_watchlist_to_dict(self):
        """Test Watchlist to_dict conversion."""
        watchlist = Watchlist(
            id=1,
            name="Test List",
            description="A test watchlist",
            items=[
                WatchlistItem(ticker="MRNA", company_name="Moderna"),
                WatchlistItem(ticker="PFE", company_name="Pfizer"),
            ],
        )

        data = watchlist.to_dict(include_items=True)

        assert data["id"] == 1
        assert data["name"] == "Test List"
        assert len(data["items"]) == 2
        assert data["items"][0]["ticker"] == "MRNA"

    def test_watchlist_from_dict(self):
        """Test Watchlist from_dict creation."""
        data = {
            "name": "Imported List",
            "description": "From export",
            "items": [
                {"ticker": "GILD", "company_name": "Gilead"},
                {"ticker": "ABBV", "company_name": "AbbVie"},
            ],
        }

        watchlist = Watchlist.from_dict(data)

        assert watchlist.name == "Imported List"
        assert len(watchlist.items) == 2
        assert watchlist.items[0].ticker == "GILD"


class TestWatchlistManager:
    """Test WatchlistManager."""

    def test_create_default_watchlist(self, manager):
        """Test that default watchlist is created."""
        watchlists = manager.get_all_watchlists()

        assert len(watchlists) >= 1
        assert any(wl.is_default for wl in watchlists)

    def test_create_watchlist(self, manager):
        """Test creating a new watchlist."""
        watchlist = manager.create_watchlist(
            name="Tech Watchlist",
            description="Tech sector plays",
        )

        assert watchlist.id is not None
        assert watchlist.name == "Tech Watchlist"
        assert watchlist.description == "Tech sector plays"

    def test_get_watchlist(self, manager):
        """Test getting a watchlist by ID."""
        created = manager.create_watchlist("Test WL")
        retrieved = manager.get_watchlist(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "Test WL"

    def test_get_default_watchlist(self, manager):
        """Test getting the default watchlist."""
        default = manager.get_default_watchlist()

        assert default is not None
        assert default.is_default is True

    def test_update_watchlist(self, manager):
        """Test updating a watchlist."""
        watchlist = manager.create_watchlist("Original Name")
        manager.update_watchlist(
            watchlist.id,
            name="Updated Name",
            description="New description",
        )

        updated = manager.get_watchlist(watchlist.id)
        assert updated.name == "Updated Name"
        assert updated.description == "New description"

    def test_delete_watchlist(self, manager):
        """Test deleting a watchlist."""
        watchlist = manager.create_watchlist("To Delete")
        deleted = manager.delete_watchlist(watchlist.id)

        assert deleted is True
        assert manager.get_watchlist(watchlist.id) is None

    def test_cannot_delete_only_watchlist(self, manager):
        """Test that the only watchlist cannot be deleted."""
        watchlists = manager.get_all_watchlists()

        # Delete all but one
        for wl in watchlists[1:]:
            manager.delete_watchlist(wl.id)

        # Try to delete the last one
        remaining = manager.get_all_watchlists()
        assert len(remaining) == 1

        deleted = manager.delete_watchlist(remaining[0].id)
        # Should fail if it's the default
        assert manager.get_all_watchlists()  # At least one remains


class TestWatchlistManagerItems:
    """Test WatchlistManager item operations."""

    def test_add_item(self, manager):
        """Test adding an item to a watchlist."""
        default = manager.get_default_watchlist()
        item = manager.add_item(
            watchlist_id=default.id,
            ticker="MRNA",
            company_name="Moderna Inc.",
            notes="Covid vaccine leader",
            price_target=175.0,
        )

        assert item is not None
        assert item.ticker == "MRNA"
        assert item.company_name == "Moderna Inc."
        assert item.price_target == 175.0

    def test_add_item_uppercase_ticker(self, manager):
        """Test that ticker is uppercased."""
        default = manager.get_default_watchlist()
        item = manager.add_item(
            watchlist_id=default.id,
            ticker="mrna",
        )

        assert item.ticker == "MRNA"

    def test_add_duplicate_item(self, manager):
        """Test adding duplicate item returns None."""
        default = manager.get_default_watchlist()
        manager.add_item(watchlist_id=default.id, ticker="PFE")
        duplicate = manager.add_item(watchlist_id=default.id, ticker="PFE")

        assert duplicate is None

    def test_get_item(self, manager):
        """Test getting an item by ID."""
        default = manager.get_default_watchlist()
        created = manager.add_item(watchlist_id=default.id, ticker="GILD")
        retrieved = manager.get_item(created.id)

        assert retrieved is not None
        assert retrieved.ticker == "GILD"

    def test_get_item_by_ticker(self, manager):
        """Test getting an item by ticker."""
        default = manager.get_default_watchlist()
        manager.add_item(watchlist_id=default.id, ticker="ABBV")
        item = manager.get_item_by_ticker(default.id, "abbv")  # lowercase

        assert item is not None
        assert item.ticker == "ABBV"

    def test_update_item(self, manager):
        """Test updating an item."""
        default = manager.get_default_watchlist()
        item = manager.add_item(watchlist_id=default.id, ticker="REGN")

        manager.update_item(
            item.id,
            notes="Updated notes",
            price_target=800.0,
            alert_enabled=True,
        )

        updated = manager.get_item(item.id)
        assert updated.notes == "Updated notes"
        assert updated.price_target == 800.0
        assert updated.alert_enabled is True

    def test_remove_item(self, manager):
        """Test removing an item."""
        default = manager.get_default_watchlist()
        item = manager.add_item(watchlist_id=default.id, ticker="BIIB")

        removed = manager.remove_item(item.id)
        assert removed is True
        assert manager.get_item(item.id) is None

    def test_remove_item_by_ticker(self, manager):
        """Test removing an item by ticker."""
        default = manager.get_default_watchlist()
        manager.add_item(watchlist_id=default.id, ticker="AMGN")

        removed = manager.remove_item_by_ticker(default.id, "AMGN")
        assert removed is True
        assert manager.get_item_by_ticker(default.id, "AMGN") is None

    def test_is_in_watchlist(self, manager):
        """Test checking if ticker is in watchlist."""
        default = manager.get_default_watchlist()
        manager.add_item(watchlist_id=default.id, ticker="VRTX")

        assert manager.is_in_watchlist(default.id, "VRTX") is True
        assert manager.is_in_watchlist(default.id, "AAPL") is False

    def test_get_tickers(self, manager):
        """Test getting all tickers in a watchlist."""
        default = manager.get_default_watchlist()
        manager.add_item(watchlist_id=default.id, ticker="MRNA")
        manager.add_item(watchlist_id=default.id, ticker="PFE")
        manager.add_item(watchlist_id=default.id, ticker="GILD")

        tickers = manager.get_tickers(default.id)
        assert "GILD" in tickers
        assert "MRNA" in tickers
        assert "PFE" in tickers

    def test_get_items_with_alerts(self, manager):
        """Test getting items with alerts enabled."""
        default = manager.get_default_watchlist()
        manager.add_item(
            watchlist_id=default.id,
            ticker="MRNA",
            alert_enabled=True,
        )
        manager.add_item(
            watchlist_id=default.id,
            ticker="PFE",
            alert_enabled=False,
        )

        alert_items = manager.get_items_with_alerts()
        tickers = [item.ticker for item in alert_items]

        assert "MRNA" in tickers
        assert "PFE" not in tickers


class TestWatchlistManagerImportExport:
    """Test import/export functionality."""

    def test_export_watchlist(self, manager):
        """Test exporting a watchlist to JSON."""
        default = manager.get_default_watchlist()
        manager.add_item(watchlist_id=default.id, ticker="MRNA")
        manager.add_item(watchlist_id=default.id, ticker="PFE")

        json_str = manager.export_watchlist(default.id)
        data = json.loads(json_str)

        assert data["name"] == "Default"
        assert len(data["items"]) == 2

    def test_import_watchlist(self, manager):
        """Test importing a watchlist from JSON."""
        json_data = json.dumps({
            "name": "Imported Portfolio",
            "description": "Test import",
            "items": [
                {"ticker": "GILD", "company_name": "Gilead"},
                {"ticker": "ABBV", "company_name": "AbbVie"},
            ],
        })

        imported = manager.import_watchlist(json_data)

        assert imported is not None
        assert imported.name == "Imported Portfolio"
        assert len(imported.items) == 2

    def test_import_watchlist_merge(self, manager):
        """Test importing with merge option."""
        default = manager.get_default_watchlist()
        manager.add_item(watchlist_id=default.id, ticker="MRNA")

        json_data = json.dumps({
            "name": "Default",  # Same name as default
            "items": [
                {"ticker": "PFE", "company_name": "Pfizer"},
            ],
        })

        imported = manager.import_watchlist(json_data, merge=True)

        # Should merge with existing Default
        tickers = manager.get_tickers(imported.id)
        assert "MRNA" in tickers
        assert "PFE" in tickers

    def test_export_all(self, manager):
        """Test exporting all watchlists."""
        manager.create_watchlist("List 1")
        manager.create_watchlist("List 2")

        json_str = manager.export_all()
        data = json.loads(json_str)

        assert "version" in data
        assert "exported_at" in data
        assert "watchlists" in data
        assert len(data["watchlists"]) >= 3  # Default + 2 created
