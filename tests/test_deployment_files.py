from pathlib import Path


def test_docker_compose_contains_required_services_and_network() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "app:" in compose
    assert "postgres:" in compose
    assert "redis:" in compose
    assert "milvus:" in compose
    assert "etcd:" in compose
    assert "minio:" in compose
    assert "prometheus:" in compose
    assert "grafana:" in compose
    assert "rag_network:" in compose
    assert "healthcheck:" in compose


def test_observability_files_exist() -> None:
    assert Path("prometheus.yml").exists()
    assert Path("grafana/provisioning/datasources/datasource.yml").exists()
    assert Path("grafana/provisioning/dashboards/dashboard.yml").exists()
    assert Path("grafana/dashboards/rag-observability.json").exists()
