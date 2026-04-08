from app.core.config import Settings


def test_settings_builds_expected_dsn() -> None:
    settings = Settings(
        APP_NAME="test-app",
        POSTGRES_USER="svc_user",
        POSTGRES_PASSWORD="secret",
        POSTGRES_HOST="db",
        POSTGRES_PORT=5433,
        POSTGRES_DB="rag_db",
        CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000",
    )

    assert settings.app_name == "test-app"
    assert settings.postgres_dsn == "postgresql+asyncpg://svc_user:secret@db:5433/rag_db"
    assert settings.postgres_dsn_safe == "postgresql+asyncpg://svc_user:***@db:5433/rag_db"
    assert settings.cors_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
