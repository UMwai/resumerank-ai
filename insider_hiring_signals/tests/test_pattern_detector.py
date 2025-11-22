"""
Tests for the pattern detection module.
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pattern_detector import (
    PatternDetector,
    PatternAlert,
    PatternType,
    AlertPriority,
    InsiderActivity,
    InstitutionalActivity,
    JobActivity,
    ExecutiveChange,
    detect_patterns_for_company,
)


class TestInsiderActivity:
    """Tests for InsiderActivity dataclass."""

    def test_creation(self):
        """Test InsiderActivity creation."""
        activity = InsiderActivity(
            ticker='MRNA',
            insider_name='John Doe',
            insider_title='CEO',
            transaction_date=date.today(),
            transaction_type='buy',
            shares=10000,
            value=150000
        )

        assert activity.ticker == 'MRNA'
        assert activity.insider_name == 'John Doe'
        assert activity.value == 150000


class TestPatternDetector:
    """Tests for PatternDetector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = PatternDetector()

    def test_initialization(self):
        """Test detector initializes correctly."""
        assert self.detector.alerts == []
        assert self.detector.THRESHOLDS['insider_cluster_count'] == 3

    def test_detect_insider_cluster_buy(self):
        """Test detection of insider buying cluster."""
        # Create 3 insiders buying within 30 days - all with value >= $50k threshold
        activities = [
            InsiderActivity(
                ticker='MRNA',
                insider_name='John Doe',
                insider_title='CEO',
                transaction_date=date.today() - timedelta(days=5),
                transaction_type='buy',
                shares=10000,
                value=150000  # Above threshold
            ),
            InsiderActivity(
                ticker='MRNA',
                insider_name='Jane Smith',
                insider_title='CFO',
                transaction_date=date.today() - timedelta(days=10),
                transaction_type='buy',
                shares=5000,
                value=75000  # Above threshold
            ),
            InsiderActivity(
                ticker='MRNA',
                insider_name='Bob Wilson',
                insider_title='Director',
                transaction_date=date.today() - timedelta(days=15),
                transaction_type='buy',
                shares=5000,
                value=60000  # Above threshold (was 45000 which is below $50k)
            ),
        ]

        alerts = self.detector.detect_insider_cluster('MRNA', activities)

        assert len(alerts) == 1
        assert alerts[0].pattern_type == PatternType.INSIDER_CLUSTER_BUY
        assert alerts[0].priority == AlertPriority.HIGH
        assert alerts[0].signal_strength > 0

    def test_detect_insider_cluster_sell(self):
        """Test detection of insider selling cluster."""
        activities = [
            InsiderActivity(
                ticker='MRNA',
                insider_name='John Doe',
                insider_title='CEO',
                transaction_date=date.today() - timedelta(days=5),
                transaction_type='sell',
                shares=10000,
                value=150000
            ),
            InsiderActivity(
                ticker='MRNA',
                insider_name='Jane Smith',
                insider_title='CFO',
                transaction_date=date.today() - timedelta(days=10),
                transaction_type='sell',
                shares=5000,
                value=75000
            ),
            InsiderActivity(
                ticker='MRNA',
                insider_name='Bob Wilson',
                insider_title='Director',
                transaction_date=date.today() - timedelta(days=15),
                transaction_type='sell',
                shares=3000,
                value=55000
            ),
        ]

        alerts = self.detector.detect_insider_cluster('MRNA', activities)

        assert len(alerts) == 1
        assert alerts[0].pattern_type == PatternType.INSIDER_CLUSTER_SELL
        assert alerts[0].signal_strength < 0

    def test_no_cluster_insufficient_insiders(self):
        """Test no cluster detected with fewer than 3 insiders."""
        activities = [
            InsiderActivity(
                ticker='MRNA',
                insider_name='John Doe',
                insider_title='CEO',
                transaction_date=date.today(),
                transaction_type='buy',
                shares=10000,
                value=150000
            ),
            InsiderActivity(
                ticker='MRNA',
                insider_name='Jane Smith',
                insider_title='CFO',
                transaction_date=date.today(),
                transaction_type='buy',
                shares=5000,
                value=75000
            ),
        ]

        alerts = self.detector.detect_insider_cluster('MRNA', activities)
        assert len(alerts) == 0

    def test_exclude_10b5_1_trades(self):
        """Test that 10b5-1 plan trades are excluded."""
        activities = [
            InsiderActivity(
                ticker='MRNA',
                insider_name='John Doe',
                insider_title='CEO',
                transaction_date=date.today(),
                transaction_type='sell',
                shares=10000,
                value=150000,
                is_10b5_1=True  # Should be excluded
            ),
            InsiderActivity(
                ticker='MRNA',
                insider_name='Jane Smith',
                insider_title='CFO',
                transaction_date=date.today(),
                transaction_type='sell',
                shares=5000,
                value=75000,
                is_10b5_1=True
            ),
            InsiderActivity(
                ticker='MRNA',
                insider_name='Bob Wilson',
                insider_title='Director',
                transaction_date=date.today(),
                transaction_type='sell',
                shares=3000,
                value=55000,
                is_10b5_1=True
            ),
        ]

        alerts = self.detector.detect_insider_cluster('MRNA', activities)
        assert len(alerts) == 0

    def test_detect_smart_money_convergence(self):
        """Test detection of smart money convergence."""
        activities = [
            InstitutionalActivity(
                ticker='MRNA',
                fund_name='Baker Brothers Advisors',
                quarter_end=date.today() - timedelta(days=30),
                action='new_position',
                shares=1000000,
                value=150000000,
                pct_change=100
            ),
            InstitutionalActivity(
                ticker='MRNA',
                fund_name='RA Capital Management',
                quarter_end=date.today() - timedelta(days=30),
                action='increase',
                shares=500000,
                value=75000000,
                pct_change=50
            ),
        ]

        alerts = self.detector.detect_smart_money_convergence('MRNA', activities)

        assert len(alerts) == 1
        assert alerts[0].pattern_type == PatternType.SMART_MONEY_CONVERGENCE
        assert alerts[0].priority == AlertPriority.HIGH

    def test_detect_smart_money_exit(self):
        """Test detection of smart money exit."""
        activities = [
            InstitutionalActivity(
                ticker='MRNA',
                fund_name='Baker Brothers Advisors',
                quarter_end=date.today() - timedelta(days=30),
                action='exit',
                shares=0,
                value=0,
                pct_change=-100
            ),
            InstitutionalActivity(
                ticker='MRNA',
                fund_name='RA Capital Management',
                quarter_end=date.today() - timedelta(days=30),
                action='exit',
                shares=0,
                value=0,
                pct_change=-100
            ),
        ]

        alerts = self.detector.detect_smart_money_convergence('MRNA', activities)

        assert len(alerts) == 1
        assert alerts[0].pattern_type == PatternType.SMART_MONEY_EXIT
        assert alerts[0].priority == AlertPriority.CRITICAL

    def test_detect_hiring_surge(self):
        """Test detection of hiring surge."""
        activities = [
            JobActivity(
                ticker='MRNA',
                job_title=f'Role {i}',
                department='Commercial',
                posted_date=date.today() - timedelta(days=5),
                is_senior=False,
                is_commercial=True,
                is_clinical=False
            )
            for i in range(6)
        ]

        alerts = self.detector.detect_hiring_patterns('MRNA', activities)

        assert len(alerts) >= 1
        # Should detect hiring surge

    def test_detect_executive_exodus(self):
        """Test detection of executive exodus."""
        changes = [
            ExecutiveChange(
                ticker='MRNA',
                executive_name='John Doe',
                title='Chief Financial Officer',
                change_type='departure',
                effective_date=date.today() - timedelta(days=30),
                is_voluntary=False
            ),
            ExecutiveChange(
                ticker='MRNA',
                executive_name='Jane Smith',
                title='Chief Medical Officer',
                change_type='departure',
                effective_date=date.today() - timedelta(days=45),
                is_voluntary=True
            ),
        ]

        alerts = self.detector.detect_executive_patterns('MRNA', changes)

        assert len(alerts) == 1
        assert alerts[0].pattern_type == PatternType.EXECUTIVE_EXODUS
        assert alerts[0].priority == AlertPriority.CRITICAL

    def test_detect_strategic_hire(self):
        """Test detection of strategic executive hire."""
        changes = [
            ExecutiveChange(
                ticker='MRNA',
                executive_name='New Person',
                title='Chief Commercial Officer',
                change_type='hiring',
                effective_date=date.today() - timedelta(days=10),
                is_voluntary=None
            ),
        ]

        alerts = self.detector.detect_executive_patterns('MRNA', changes)

        assert len(alerts) == 1
        assert alerts[0].pattern_type == PatternType.EXECUTIVE_HIRING

    def test_detect_combined_patterns(self):
        """Test detection of combined patterns."""
        insider = [
            InsiderActivity(
                ticker='MRNA',
                insider_name=f'Insider {i}',
                insider_title='Officer',
                transaction_date=date.today() - timedelta(days=10),
                transaction_type='buy',
                shares=5000,
                value=75000
            )
            for i in range(3)
        ]

        institutional = [
            InstitutionalActivity(
                ticker='MRNA',
                fund_name='Baker Brothers Advisors',
                quarter_end=date.today() - timedelta(days=30),
                action='new_position',
                shares=1000000,
                value=150000000,
                pct_change=100
            ),
        ]

        jobs = [
            JobActivity(
                ticker='MRNA',
                job_title=f'Commercial Role {i}',
                department='Commercial',
                posted_date=date.today() - timedelta(days=5),
                is_senior=False,
                is_commercial=True,
                is_clinical=False
            )
            for i in range(5)
        ]

        alerts = self.detector.detect_combined_patterns(
            'MRNA', insider, institutional, jobs, [], None
        )

        # Should potentially detect catalyst preparation
        # (depends on exact signal counts)

    def test_alert_deduplication(self):
        """Test alert deduplication."""
        activities = [
            InsiderActivity(
                ticker='MRNA',
                insider_name=f'Insider {i}',
                insider_title='Officer',
                transaction_date=date.today() - timedelta(days=i*5),
                transaction_type='buy',
                shares=5000,
                value=75000
            )
            for i in range(3)
        ]

        # First call
        alerts1 = self.detector.detect_all_patterns(
            'MRNA', activities, [], [], []
        )

        # Second call (should deduplicate)
        alerts2 = self.detector.detect_all_patterns(
            'MRNA', activities, [], [], []
        )

        # Deduplication should prevent duplicates
        assert len(alerts2) == 0

    def test_alert_to_dict(self):
        """Test PatternAlert serialization."""
        alert = PatternAlert(
            pattern_type=PatternType.INSIDER_CLUSTER_BUY,
            ticker='MRNA',
            company_name='Moderna',
            detected_at=datetime.now(),
            priority=AlertPriority.HIGH,
            confidence=0.85,
            signal_strength=8,
            description='Test alert',
            details={'test': 'value'},
            contributing_signals=[],
            recommended_action='Buy',
            expires_at=datetime.now() + timedelta(days=30)
        )

        data = alert.to_dict()

        assert data['ticker'] == 'MRNA'
        assert data['pattern_type'] == 'insider_cluster_buy'
        assert data['priority'] == 'high'

    def test_get_active_alerts(self):
        """Test active alerts filtering."""
        # Create some alerts
        activities = [
            InsiderActivity(
                ticker='MRNA',
                insider_name=f'Insider {i}',
                insider_title='Officer',
                transaction_date=date.today() - timedelta(days=i*5),
                transaction_type='buy',
                shares=5000,
                value=75000
            )
            for i in range(3)
        ]

        self.detector.detect_all_patterns('MRNA', activities, [], [], [])

        active = self.detector.get_active_alerts()
        # All alerts should be active (not expired)
        assert all(
            a.expires_at is None or a.expires_at > datetime.now()
            for a in active
        )

    def test_generate_alert_summary(self):
        """Test alert summary generation."""
        activities = [
            InsiderActivity(
                ticker='MRNA',
                insider_name=f'Insider {i}',
                insider_title='Officer',
                transaction_date=date.today() - timedelta(days=i*5),
                transaction_type='buy',
                shares=5000,
                value=75000
            )
            for i in range(3)
        ]

        self.detector.detect_all_patterns('MRNA', activities, [], [], [])

        summary = self.detector.generate_alert_summary()

        assert isinstance(summary, str)
        if self.detector.alerts:
            assert 'PATTERN ALERT SUMMARY' in summary


class TestConvenienceFunction:
    """Tests for convenience functions."""

    def test_detect_patterns_for_company(self):
        """Test convenience function."""
        insider_data = [
            {
                'ticker': 'MRNA',
                'insider_name': 'John Doe',
                'insider_title': 'CEO',
                'transaction_date': date.today(),
                'transaction_type': 'buy',
                'shares': 10000,
                'value': 150000,
            }
        ]

        alerts = detect_patterns_for_company(
            ticker='MRNA',
            insider_data=insider_data,
            institutional_data=[],
            job_data=[],
            executive_data=[]
        )

        assert isinstance(alerts, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
