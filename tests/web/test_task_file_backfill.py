"""Test task file backfill API functionality"""

import os
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.auth import hash_password
from xagent.web.api.files import file_router
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.database import Base, get_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User

# Global test session storage
_test_session_local = None


@pytest.fixture(scope="function")
def test_db():
    """Create test database with isolated engine and session"""
    global _test_session_local

    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)

    test_engine = create_engine(
        f"sqlite:///{temp_db_path}", connect_args={"check_same_thread": False}
    )
    _test_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    def override_get_db():
        db = _test_session_local()
        try:
            yield db
        finally:
            db.close()

    test_app = FastAPI()
    test_app.include_router(file_router)
    test_app.dependency_overrides[get_db] = override_get_db

    Base.metadata.create_all(bind=test_engine)

    session = _test_session_local()
    try:
        admin_user = User(
            username="admin", password_hash=hash_password("admin"), is_admin=True
        )
        regular_user = User(
            username="user", password_hash=hash_password("user"), is_admin=False
        )
        session.add(admin_user)
        session.add(regular_user)
        session.commit()
        session.refresh(admin_user)
        session.refresh(regular_user)
        yield admin_user, regular_user, test_app, session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()
        try:
            os.unlink(temp_db_path)
        except OSError:
            pass


@pytest.fixture(scope="function")
def auth_headers(test_db):
    """Authentication headers for admin user"""
    admin_user, _, _, _ = test_db
    from datetime import datetime, timedelta

    import jwt

    payload = {
        "sub": admin_user.username,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "user_id": admin_user.id,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def user_auth_headers(test_db):
    """Authentication headers for regular user"""
    _, regular_user, _, _ = test_db
    from datetime import datetime, timedelta

    import jwt

    payload = {
        "sub": regular_user.username,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "user_id": regular_user.id,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def client(test_db):
    """Create test client"""
    _, _, test_app, _ = test_db
    return TestClient(test_app)


@pytest.fixture(scope="function")
def temp_uploads_dir(monkeypatch):
    """Create temporary uploads directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        import xagent.web.api.files
        import xagent.web.config

        monkeypatch.setattr(xagent.web.config, "UPLOADS_DIR", temp_path)
        monkeypatch.setattr(xagent.web.api.files, "UPLOADS_DIR", temp_path)

        yield temp_path


class TestTaskFileBackfill:
    """Test task file backfill functionality"""

    def test_backfill_task_files_success(
        self, client, test_db, temp_uploads_dir, user_auth_headers
    ):
        """Test successful backfill of files for a task"""
        _, regular_user, _, session = test_db

        # Create a task in database
        task = Task(
            user_id=regular_user.id,
            title="test task",
            status=TaskStatus.COMPLETED,
            model_name="test-model",
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

        # Create files on disk WITHOUT database records (simulating AI-generated files)
        user_root = temp_uploads_dir / f"user_{regular_user.id}"
        task_dir = user_root / f"web_task_{task_id}"
        task_dir.mkdir(parents=True)

        # Create test files
        (task_dir / "chart.html").write_text("<html><body>Chart</body></html>")
        (task_dir / "data.json").write_text('{"data": [1,2,3]}')
        (task_dir / "output.txt").write_text("Output from AI")

        # Call backfill endpoint
        response = client.post(
            f"/api/files/task/{task_id}/backfill",
            headers=user_auth_headers,
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task_id"] == task_id
        assert data["count"] == 3

        # Verify files are now in database
        files = (
            session.query(UploadedFile).filter(UploadedFile.task_id == task_id).all()
        )
        assert len(files) == 3

        filenames = {f.filename for f in files}
        assert "chart.html" in filenames
        assert "data.json" in filenames
        assert "output.txt" in filenames

        # Verify file metadata
        for f in files:
            assert f.task_id == task_id
            assert f.user_id == regular_user.id
            assert f.mime_type is not None
            assert f.file_size > 0

    def test_backfill_task_files_idempotent(
        self, client, test_db, temp_uploads_dir, user_auth_headers
    ):
        """Test that backfill is idempotent (safe to call multiple times)"""
        _, regular_user, _, session = test_db

        task = Task(
            user_id=regular_user.id,
            title="test task",
            status=TaskStatus.COMPLETED,
            model_name="test-model",
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

        # Create files on disk
        user_root = temp_uploads_dir / f"user_{regular_user.id}"
        task_dir = user_root / f"web_task_{task_id}"
        task_dir.mkdir(parents=True)
        (task_dir / "test.html").write_text("<html>test</html>")

        # First backfill
        response1 = client.post(
            f"/api/files/task/{task_id}/backfill",
            headers=user_auth_headers,
        )
        assert response1.status_code == 200
        assert response1.json()["count"] == 1

        # Second backfill (should not create duplicates)
        response2 = client.post(
            f"/api/files/task/{task_id}/backfill",
            headers=user_auth_headers,
        )
        assert response2.status_code == 200
        assert response2.json()["count"] == 0

        # Verify only one record exists
        files = (
            session.query(UploadedFile).filter(UploadedFile.task_id == task_id).all()
        )
        assert len(files) == 1

    def test_backfill_task_not_found(self, client, user_auth_headers):
        """Test backfill with non-existent task"""
        response = client.post(
            "/api/files/task/99999/backfill",
            headers=user_auth_headers,
        )
        assert response.status_code == 404

    def test_backfill_task_unauthorized_other_user(
        self, client, test_db, temp_uploads_dir, user_auth_headers
    ):
        """Test that user cannot backfill another user's task"""
        admin_user, regular_user, _, session = test_db

        # Create task for admin user
        task = Task(
            user_id=admin_user.id,  # Different from regular_user
            title="admin task",
            status=TaskStatus.COMPLETED,
            model_name="test-model",
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

        # Try to backfill as regular user (should be denied)
        response = client.post(
            f"/api/files/task/{task_id}/backfill",
            headers=user_auth_headers,  # Regular user, not admin
        )
        assert response.status_code == 403

    def test_backfill_task_with_existing_files(
        self, client, test_db, temp_uploads_dir, user_auth_headers
    ):
        """Test backfill when some files are already registered"""
        _, regular_user, _, session = test_db

        task = Task(
            user_id=regular_user.id,
            title="test task",
            status=TaskStatus.COMPLETED,
            model_name="test-model",
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

        user_root = temp_uploads_dir / f"user_{regular_user.id}"
        task_dir = user_root / f"web_task_{task_id}"
        task_dir.mkdir(parents=True)

        # Create one file and register it in database
        existing_file = task_dir / "existing.txt"
        existing_file.write_text("Already registered")
        existing_record = UploadedFile(
            user_id=regular_user.id,
            task_id=task_id,
            filename="existing.txt",
            storage_path=str(existing_file),
            mime_type="text/plain",
            file_size=17,
        )
        session.add(existing_record)
        session.commit()

        # Create new file NOT in database
        new_file = task_dir / "new.html"
        new_file.write_text("<html>New file</html>")

        # Backfill should only register the new file
        response = client.post(
            f"/api/files/task/{task_id}/backfill",
            headers=user_auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["count"] == 1

        # Verify total files
        files = (
            session.query(UploadedFile).filter(UploadedFile.task_id == task_id).all()
        )
        assert len(files) == 2

        filenames = {f.filename for f in files}
        assert "existing.txt" in filenames
        assert "new.html" in filenames

    def test_backfill_task_naming_convention_task(
        self, client, test_db, temp_uploads_dir, user_auth_headers
    ):
        """Test backfill with 'task_' directory naming convention"""
        _, regular_user, _, session = test_db

        task = Task(
            user_id=regular_user.id,
            title="test task",
            status=TaskStatus.COMPLETED,
            model_name="test-model",
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

        # Create 'task_' directory (not 'web_task_')
        user_root = temp_uploads_dir / f"user_{regular_user.id}"
        task_dir = user_root / f"task_{task_id}"
        task_dir.mkdir(parents=True)
        (task_dir / "test.txt").write_text("Test content")

        response = client.post(
            f"/api/files/task/{task_id}/backfill",
            headers=user_auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["count"] == 1
