"""
Tests for Airflow DAG validation.

Tests:
- DAG file syntax validation
- Task dependency structure
- Default args configuration
- Schedule interval validation
- Retry logic configuration
"""

import pytest
import os
import sys
from datetime import timedelta

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Airflow Variable before importing DAGs
from unittest.mock import MagicMock, patch


class MockVariable:
    """Mock Airflow Variable for testing."""

    @staticmethod
    def get(key, default_var=None):
        defaults = {
            "PROJECT_ROOT": "/opt/airflow/project",
            "ALERT_EMAIL_RECIPIENTS": "test@example.com",
        }
        return defaults.get(key, default_var)


@pytest.fixture(autouse=True)
def mock_airflow_variable():
    """Mock Airflow Variable for all tests."""
    with patch.dict('sys.modules', {'airflow.models': MagicMock()}):
        from airflow import models
        models.Variable = MockVariable
        yield


class TestDAGFileValidation:
    """Test that DAG files have valid Python syntax."""

    def test_clinical_trial_dag_syntax(self):
        """Test clinical_trial_dag.py syntax."""
        dag_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
            "clinical_trial_dag.py",
        )
        assert os.path.exists(dag_path), f"DAG file not found: {dag_path}"

        # Check file is valid Python
        with open(dag_path, "r") as f:
            source = f.read()

        try:
            compile(source, dag_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {dag_path}: {e}")

    def test_patent_ip_dag_syntax(self):
        """Test patent_ip_dag.py syntax."""
        dag_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
            "patent_ip_dag.py",
        )
        assert os.path.exists(dag_path), f"DAG file not found: {dag_path}"

        with open(dag_path, "r") as f:
            source = f.read()

        try:
            compile(source, dag_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {dag_path}: {e}")

    def test_insider_hiring_dag_syntax(self):
        """Test insider_hiring_dag.py syntax."""
        dag_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
            "insider_hiring_dag.py",
        )
        assert os.path.exists(dag_path), f"DAG file not found: {dag_path}"

        with open(dag_path, "r") as f:
            source = f.read()

        try:
            compile(source, dag_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {dag_path}: {e}")


class TestDAGConfiguration:
    """Test DAG configuration values."""

    def test_dag_default_args_structure(self):
        """Test that DAGs have required default_args fields."""
        required_fields = [
            "owner",
            "depends_on_past",
            "email_on_failure",
            "retries",
            "retry_delay",
        ]

        # Read DAG files and check for required fields
        dag_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
        )

        dag_files = [
            "clinical_trial_dag.py",
            "patent_ip_dag.py",
            "insider_hiring_dag.py",
        ]

        for dag_file in dag_files:
            dag_path = os.path.join(dag_dir, dag_file)
            with open(dag_path, "r") as f:
                content = f.read()

            for field in required_fields:
                assert f'"{field}"' in content or f"'{field}'" in content, \
                    f"Missing {field} in default_args for {dag_file}"

    def test_dag_has_retry_exponential_backoff(self):
        """Test that DAGs have exponential backoff configured."""
        dag_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
        )

        dag_files = [
            "clinical_trial_dag.py",
            "patent_ip_dag.py",
        ]

        for dag_file in dag_files:
            dag_path = os.path.join(dag_dir, dag_file)
            with open(dag_path, "r") as f:
                content = f.read()

            assert "retry_exponential_backoff" in content, \
                f"Missing retry_exponential_backoff in {dag_file}"

    def test_dag_has_max_retry_delay(self):
        """Test that DAGs have max_retry_delay configured."""
        dag_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
        )

        for dag_file in ["clinical_trial_dag.py", "patent_ip_dag.py"]:
            dag_path = os.path.join(dag_dir, dag_file)
            with open(dag_path, "r") as f:
                content = f.read()

            assert "max_retry_delay" in content, \
                f"Missing max_retry_delay in {dag_file}"


class TestScheduleValidation:
    """Test DAG schedule configurations."""

    def test_clinical_trial_schedule(self):
        """Test clinical trial DAG runs daily at 6 PM ET."""
        dag_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
            "clinical_trial_dag.py",
        )

        with open(dag_path, "r") as f:
            content = f.read()

        # Should run at 6 PM (18:00)
        assert "0 18 * * *" in content, \
            "Clinical trial DAG should run at 6 PM daily"

    def test_patent_ip_schedule(self):
        """Test patent IP DAG runs weekly on Monday."""
        dag_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
            "patent_ip_dag.py",
        )

        with open(dag_path, "r") as f:
            content = f.read()

        # Should run on Monday at 8 AM
        assert "0 8 * * 1" in content, \
            "Patent IP DAG should run Monday at 8 AM"


class TestOperatorFileValidation:
    """Test custom operator file validation."""

    def test_operators_init_file_exists(self):
        """Test operators __init__.py exists."""
        init_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
            "operators",
            "__init__.py",
        )
        assert os.path.exists(init_path), "operators/__init__.py not found"

    def test_data_operators_syntax(self):
        """Test data_operators.py syntax."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
            "operators",
            "data_operators.py",
        )

        if os.path.exists(path):
            with open(path, "r") as f:
                source = f.read()

            try:
                compile(source, path, "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in data_operators.py: {e}")

    def test_signal_operators_syntax(self):
        """Test signal_operators.py syntax."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
            "operators",
            "signal_operators.py",
        )

        if os.path.exists(path):
            with open(path, "r") as f:
                source = f.read()

            try:
                compile(source, path, "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in signal_operators.py: {e}")

    def test_alert_operators_syntax(self):
        """Test alert_operators.py syntax."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dags",
            "operators",
            "alert_operators.py",
        )

        if os.path.exists(path):
            with open(path, "r") as f:
                source = f.read()

            try:
                compile(source, path, "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in alert_operators.py: {e}")


class TestDockerComposeValidation:
    """Test Docker Compose configuration files."""

    def test_docker_compose_yaml_valid(self):
        """Test docker-compose.yml is valid YAML."""
        import yaml

        compose_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docker",
            "docker-compose.yml",
        )

        with open(compose_path, "r") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in docker-compose.yml: {e}")

        assert "services" in config, "docker-compose.yml missing services section"

    def test_docker_compose_dev_yaml_valid(self):
        """Test docker-compose.dev.yml is valid YAML."""
        import yaml

        compose_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docker",
            "docker-compose.dev.yml",
        )

        if os.path.exists(compose_path):
            with open(compose_path, "r") as f:
                try:
                    config = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in docker-compose.dev.yml: {e}")

            assert "services" in config, "docker-compose.dev.yml missing services section"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
