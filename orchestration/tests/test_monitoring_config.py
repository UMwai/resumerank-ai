"""
Tests for Monitoring Configuration

This module tests:
- Alertmanager configuration validity
- Alert rules syntax and logic
- Grafana dashboard validity
- Loki configuration
- Prometheus configuration
"""

import json
import os
import unittest
from pathlib import Path
import yaml


# Base paths
BASE_DIR = Path(__file__).parent.parent
DOCKER_DIR = BASE_DIR / "docker"
GRAFANA_DIR = DOCKER_DIR / "grafana"


class TestPrometheusConfig(unittest.TestCase):
    """Test Prometheus configuration."""

    def setUp(self):
        self.prometheus_config_path = DOCKER_DIR / "prometheus.yml"

    def test_prometheus_config_exists(self):
        """Test that prometheus.yml exists."""
        self.assertTrue(
            self.prometheus_config_path.exists(),
            f"Prometheus config not found at {self.prometheus_config_path}"
        )

    def test_prometheus_config_valid_yaml(self):
        """Test that prometheus.yml is valid YAML."""
        with open(self.prometheus_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config, dict)

    def test_prometheus_has_scrape_configs(self):
        """Test that Prometheus has scrape configurations."""
        with open(self.prometheus_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("scrape_configs", config)
        self.assertIsInstance(config["scrape_configs"], list)
        self.assertGreater(len(config["scrape_configs"]), 0)

    def test_prometheus_alertmanager_configured(self):
        """Test that Alertmanager is configured in Prometheus."""
        with open(self.prometheus_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("alerting", config)
        self.assertIn("alertmanagers", config["alerting"])

    def test_prometheus_rule_files_configured(self):
        """Test that rule files are configured."""
        with open(self.prometheus_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("rule_files", config)


class TestAlertRules(unittest.TestCase):
    """Test Prometheus alert rules."""

    def setUp(self):
        self.alert_rules_path = DOCKER_DIR / "alert_rules.yml"

    def test_alert_rules_exists(self):
        """Test that alert_rules.yml exists."""
        self.assertTrue(
            self.alert_rules_path.exists(),
            f"Alert rules not found at {self.alert_rules_path}"
        )

    def test_alert_rules_valid_yaml(self):
        """Test that alert_rules.yml is valid YAML."""
        with open(self.alert_rules_path) as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config, dict)

    def test_alert_rules_has_groups(self):
        """Test that alert rules has groups."""
        with open(self.alert_rules_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("groups", config)
        self.assertIsInstance(config["groups"], list)
        self.assertGreater(len(config["groups"]), 0)

    def test_alert_rules_have_required_fields(self):
        """Test that each alert rule has required fields."""
        with open(self.alert_rules_path) as f:
            config = yaml.safe_load(f)

        required_fields = ["alert", "expr", "labels", "annotations"]

        for group in config["groups"]:
            self.assertIn("name", group)
            self.assertIn("rules", group)

            for rule in group["rules"]:
                for field in required_fields:
                    self.assertIn(
                        field, rule,
                        f"Rule missing required field '{field}': {rule.get('alert', 'unknown')}"
                    )

    def test_alert_rules_have_severity(self):
        """Test that each alert has a severity label."""
        with open(self.alert_rules_path) as f:
            config = yaml.safe_load(f)

        for group in config["groups"]:
            for rule in group["rules"]:
                self.assertIn(
                    "severity", rule.get("labels", {}),
                    f"Rule missing severity label: {rule.get('alert', 'unknown')}"
                )

    def test_alert_rules_have_runbook_url(self):
        """Test that each alert has a runbook URL."""
        with open(self.alert_rules_path) as f:
            config = yaml.safe_load(f)

        for group in config["groups"]:
            for rule in group["rules"]:
                self.assertIn(
                    "runbook_url", rule.get("annotations", {}),
                    f"Rule missing runbook_url: {rule.get('alert', 'unknown')}"
                )

    def test_severity_levels_valid(self):
        """Test that severity levels are valid."""
        with open(self.alert_rules_path) as f:
            config = yaml.safe_load(f)

        valid_severities = {"critical", "warning", "info"}

        for group in config["groups"]:
            for rule in group["rules"]:
                severity = rule.get("labels", {}).get("severity")
                self.assertIn(
                    severity, valid_severities,
                    f"Invalid severity '{severity}' for rule: {rule.get('alert', 'unknown')}"
                )


class TestAlertmanagerConfig(unittest.TestCase):
    """Test Alertmanager configuration."""

    def setUp(self):
        self.alertmanager_config_path = DOCKER_DIR / "alertmanager.yml"

    def test_alertmanager_config_exists(self):
        """Test that alertmanager.yml exists."""
        self.assertTrue(
            self.alertmanager_config_path.exists(),
            f"Alertmanager config not found at {self.alertmanager_config_path}"
        )

    def test_alertmanager_config_valid_yaml(self):
        """Test that alertmanager.yml is valid YAML."""
        with open(self.alertmanager_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config, dict)

    def test_alertmanager_has_route(self):
        """Test that Alertmanager has route configuration."""
        with open(self.alertmanager_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("route", config)

    def test_alertmanager_has_receivers(self):
        """Test that Alertmanager has receivers."""
        with open(self.alertmanager_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("receivers", config)
        self.assertIsInstance(config["receivers"], list)
        self.assertGreater(len(config["receivers"]), 0)

    def test_alertmanager_receivers_have_names(self):
        """Test that all receivers have names."""
        with open(self.alertmanager_config_path) as f:
            config = yaml.safe_load(f)

        for receiver in config["receivers"]:
            self.assertIn("name", receiver)

    def test_alertmanager_default_receiver_exists(self):
        """Test that the default receiver referenced in route exists."""
        with open(self.alertmanager_config_path) as f:
            config = yaml.safe_load(f)

        default_receiver = config["route"]["receiver"]
        receiver_names = [r["name"] for r in config["receivers"]]

        self.assertIn(
            default_receiver, receiver_names,
            f"Default receiver '{default_receiver}' not found in receivers"
        )

    def test_alertmanager_has_inhibit_rules(self):
        """Test that Alertmanager has inhibit rules."""
        with open(self.alertmanager_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("inhibit_rules", config)
        self.assertIsInstance(config["inhibit_rules"], list)

    def test_alertmanager_has_critical_receiver(self):
        """Test that a critical alerts receiver exists."""
        with open(self.alertmanager_config_path) as f:
            config = yaml.safe_load(f)

        receiver_names = [r["name"] for r in config["receivers"]]
        has_critical = any("critical" in name.lower() for name in receiver_names)

        self.assertTrue(has_critical, "No critical alerts receiver found")

    def test_alertmanager_has_cost_receiver(self):
        """Test that a cost alerts receiver exists."""
        with open(self.alertmanager_config_path) as f:
            config = yaml.safe_load(f)

        receiver_names = [r["name"] for r in config["receivers"]]
        has_cost = any("cost" in name.lower() for name in receiver_names)

        self.assertTrue(has_cost, "No cost alerts receiver found")


class TestLokiConfig(unittest.TestCase):
    """Test Loki configuration."""

    def setUp(self):
        self.loki_config_path = DOCKER_DIR / "loki-config.yml"

    def test_loki_config_exists(self):
        """Test that loki-config.yml exists."""
        self.assertTrue(
            self.loki_config_path.exists(),
            f"Loki config not found at {self.loki_config_path}"
        )

    def test_loki_config_valid_yaml(self):
        """Test that loki-config.yml is valid YAML."""
        with open(self.loki_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config, dict)

    def test_loki_has_retention_config(self):
        """Test that Loki has retention configuration."""
        with open(self.loki_config_path) as f:
            config = yaml.safe_load(f)

        # Check for retention in limits_config or compactor
        has_retention = (
            "retention_period" in config.get("limits_config", {}) or
            "retention_enabled" in config.get("compactor", {})
        )

        self.assertTrue(has_retention, "No retention configuration found")

    def test_loki_retention_is_30_days(self):
        """Test that Loki retention is 30 days (720h)."""
        with open(self.loki_config_path) as f:
            config = yaml.safe_load(f)

        retention = config.get("limits_config", {}).get("retention_period", "")

        self.assertEqual(retention, "720h", "Retention should be 720h (30 days)")


class TestPromtailConfig(unittest.TestCase):
    """Test Promtail configuration."""

    def setUp(self):
        self.promtail_config_path = DOCKER_DIR / "promtail-config.yml"

    def test_promtail_config_exists(self):
        """Test that promtail-config.yml exists."""
        self.assertTrue(
            self.promtail_config_path.exists(),
            f"Promtail config not found at {self.promtail_config_path}"
        )

    def test_promtail_config_valid_yaml(self):
        """Test that promtail-config.yml is valid YAML."""
        with open(self.promtail_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config, dict)

    def test_promtail_has_clients(self):
        """Test that Promtail has client configuration."""
        with open(self.promtail_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("clients", config)
        self.assertGreater(len(config["clients"]), 0)

    def test_promtail_has_scrape_configs(self):
        """Test that Promtail has scrape configurations."""
        with open(self.promtail_config_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("scrape_configs", config)
        self.assertGreater(len(config["scrape_configs"]), 0)


class TestGrafanaDashboards(unittest.TestCase):
    """Test Grafana dashboards."""

    def setUp(self):
        self.dashboards_dir = GRAFANA_DIR / "dashboards"
        self.required_dashboards = [
            "pipeline-overview.json",
            "cost-monitoring.json",
            "alert-history.json",
            "sla-compliance.json",
            "data-quality.json"
        ]

    def test_dashboards_directory_exists(self):
        """Test that dashboards directory exists."""
        self.assertTrue(
            self.dashboards_dir.exists(),
            f"Dashboards directory not found at {self.dashboards_dir}"
        )

    def test_required_dashboards_exist(self):
        """Test that required dashboards exist."""
        for dashboard in self.required_dashboards:
            dashboard_path = self.dashboards_dir / dashboard
            self.assertTrue(
                dashboard_path.exists(),
                f"Required dashboard not found: {dashboard}"
            )

    def test_dashboards_valid_json(self):
        """Test that all dashboards are valid JSON."""
        for dashboard_file in self.dashboards_dir.glob("*.json"):
            with open(dashboard_file) as f:
                try:
                    dashboard = json.load(f)
                    self.assertIsInstance(dashboard, dict)
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON in {dashboard_file.name}: {e}")

    def test_dashboards_have_required_fields(self):
        """Test that dashboards have required fields."""
        required_fields = ["title", "uid", "panels"]

        for dashboard_file in self.dashboards_dir.glob("*.json"):
            with open(dashboard_file) as f:
                dashboard = json.load(f)

            for field in required_fields:
                self.assertIn(
                    field, dashboard,
                    f"Dashboard {dashboard_file.name} missing required field '{field}'"
                )

    def test_cost_dashboard_has_budget_panels(self):
        """Test that cost dashboard has budget-related panels."""
        cost_dashboard_path = self.dashboards_dir / "cost-monitoring.json"

        if not cost_dashboard_path.exists():
            self.skipTest("Cost monitoring dashboard not found")

        with open(cost_dashboard_path) as f:
            dashboard = json.load(f)

        panel_titles = [p.get("title", "") for p in dashboard.get("panels", [])]
        panel_titles_lower = [t.lower() for t in panel_titles]

        has_budget = any("budget" in t or "cost" in t for t in panel_titles_lower)
        self.assertTrue(has_budget, "Cost dashboard should have budget-related panels")

    def test_alert_history_dashboard_has_resolution_metrics(self):
        """Test that alert history dashboard has resolution time metrics."""
        alert_dashboard_path = self.dashboards_dir / "alert-history.json"

        if not alert_dashboard_path.exists():
            self.skipTest("Alert history dashboard not found")

        with open(alert_dashboard_path) as f:
            dashboard = json.load(f)

        panel_titles = [p.get("title", "") for p in dashboard.get("panels", [])]
        panel_titles_lower = [t.lower() for t in panel_titles]

        has_resolution = any("resolution" in t for t in panel_titles_lower)
        self.assertTrue(has_resolution, "Alert dashboard should have resolution time metrics")


class TestGrafanaDatasources(unittest.TestCase):
    """Test Grafana datasources configuration."""

    def setUp(self):
        self.datasources_path = GRAFANA_DIR / "provisioning" / "datasources" / "datasources.yml"

    def test_datasources_exists(self):
        """Test that datasources.yml exists."""
        self.assertTrue(
            self.datasources_path.exists(),
            f"Datasources config not found at {self.datasources_path}"
        )

    def test_datasources_valid_yaml(self):
        """Test that datasources.yml is valid YAML."""
        with open(self.datasources_path) as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config, dict)

    def test_datasources_has_prometheus(self):
        """Test that Prometheus datasource is configured."""
        with open(self.datasources_path) as f:
            config = yaml.safe_load(f)

        datasource_names = [ds.get("name", "") for ds in config.get("datasources", [])]
        self.assertIn("Prometheus", datasource_names)

    def test_datasources_has_loki(self):
        """Test that Loki datasource is configured."""
        with open(self.datasources_path) as f:
            config = yaml.safe_load(f)

        datasource_names = [ds.get("name", "") for ds in config.get("datasources", [])]
        self.assertIn("Loki", datasource_names)

    def test_datasources_has_alertmanager(self):
        """Test that Alertmanager datasource is configured."""
        with open(self.datasources_path) as f:
            config = yaml.safe_load(f)

        datasource_names = [ds.get("name", "") for ds in config.get("datasources", [])]
        self.assertIn("Alertmanager", datasource_names)


class TestDockerCompose(unittest.TestCase):
    """Test docker-compose configuration."""

    def setUp(self):
        self.compose_path = DOCKER_DIR / "docker-compose.yml"

    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        self.assertTrue(
            self.compose_path.exists(),
            f"Docker Compose file not found at {self.compose_path}"
        )

    def test_docker_compose_valid_yaml(self):
        """Test that docker-compose.yml is valid YAML."""
        with open(self.compose_path) as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config, dict)

    def test_docker_compose_has_services(self):
        """Test that docker-compose has services."""
        with open(self.compose_path) as f:
            config = yaml.safe_load(f)

        self.assertIn("services", config)
        self.assertIsInstance(config["services"], dict)

    def test_monitoring_services_present(self):
        """Test that monitoring services are present."""
        with open(self.compose_path) as f:
            config = yaml.safe_load(f)

        required_services = [
            "prometheus",
            "alertmanager",
            "grafana",
            "loki",
            "promtail"
        ]

        services = config.get("services", {}).keys()

        for service in required_services:
            self.assertIn(
                service, services,
                f"Required service '{service}' not found in docker-compose.yml"
            )

    def test_services_have_healthchecks(self):
        """Test that key services have healthchecks."""
        with open(self.compose_path) as f:
            config = yaml.safe_load(f)

        services_requiring_healthcheck = [
            "prometheus",
            "alertmanager",
            "grafana",
            "loki"
        ]

        services = config.get("services", {})

        for service_name in services_requiring_healthcheck:
            service = services.get(service_name, {})
            self.assertIn(
                "healthcheck", service,
                f"Service '{service_name}' should have a healthcheck"
            )


class TestDeploymentScripts(unittest.TestCase):
    """Test deployment scripts."""

    def setUp(self):
        self.scripts_dir = BASE_DIR / "scripts"

    def test_aws_deploy_script_exists(self):
        """Test that AWS deployment script exists."""
        script_path = self.scripts_dir / "deploy_aws.sh"
        self.assertTrue(
            script_path.exists(),
            f"AWS deployment script not found at {script_path}"
        )

    def test_gcp_deploy_script_exists(self):
        """Test that GCP deployment script exists."""
        script_path = self.scripts_dir / "deploy_gcp.sh"
        self.assertTrue(
            script_path.exists(),
            f"GCP deployment script not found at {script_path}"
        )

    def test_scripts_are_executable(self):
        """Test that deployment scripts are executable."""
        for script_name in ["deploy_aws.sh", "deploy_gcp.sh"]:
            script_path = self.scripts_dir / script_name
            if script_path.exists():
                self.assertTrue(
                    os.access(script_path, os.X_OK),
                    f"Script {script_name} should be executable"
                )


if __name__ == "__main__":
    unittest.main()
