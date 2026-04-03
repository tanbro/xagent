"""Unit tests for core/config.py configuration functions."""

import os
import tempfile
from pathlib import Path

from xagent.config import (
    BOXLITE_HOME_DIR,
    DATABASE_URL,
    EXTERNAL_SKILLS_LIBRARY_DIRS,
    EXTERNAL_UPLOAD_DIRS,
    LANCEDB_PATH,
    SANDBOX_CPUS,
    SANDBOX_ENV,
    SANDBOX_IMAGE,
    SANDBOX_MEMORY,
    SANDBOX_VOLUMES,
    STORAGE_ROOT,
    UPLOADS_DIR,
    WEB_DIR,
    WEB_STATIC_DIR,
    get_boxlite_home_dir,
    get_database_url,
    get_default_sqlite_db_path,
    get_default_uploads_dir,
    get_external_skills_dirs,
    get_external_upload_dirs,
    get_lancedb_path,
    get_sandbox_cpus,
    get_sandbox_env,
    get_sandbox_image,
    get_sandbox_memory,
    get_sandbox_volumes,
    get_storage_root,
    get_uploads_dir,
    get_web_dir,
    get_web_static_dir,
)


class TestEnvironmentVariableConstants:
    """Test environment variable constant names."""

    def test_upload_dir_constant(self):
        assert UPLOADS_DIR == "XAGENT_UPLOADS_DIR"

    def test_web_static_dir_constant(self):
        assert WEB_STATIC_DIR == "XAGENT_WEB_STATIC_DIR"

    def test_web_dir_constant(self):
        assert WEB_DIR == "XAGENT_WEB_DIR"

    def test_external_upload_dirs_constant(self):
        assert EXTERNAL_UPLOAD_DIRS == "XAGENT_EXTERNAL_UPLOAD_DIRS"

    def test_external_skills_dirs_constant(self):
        assert EXTERNAL_SKILLS_LIBRARY_DIRS == "XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS"

    def test_storage_root_constant(self):
        assert STORAGE_ROOT == "XAGENT_STORAGE_ROOT"

    def test_sandbox_image_constant(self):
        assert SANDBOX_IMAGE == "SANDBOX_IMAGE"

    def test_lancedb_path_constant(self):
        assert LANCEDB_PATH == "LANCEDB_PATH"

    def test_database_url_constant(self):
        assert DATABASE_URL == "DATABASE_URL"


class TestGetUploadsDir:
    """Test get_uploads_dir() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(UPLOADS_DIR, None)
        os.environ.pop(WEB_DIR, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_uploads_dir(self):
        """Test default uploads directory path."""
        result = get_uploads_dir()
        # Default is src/xagent/web/uploads
        assert result.name == "uploads"
        assert result.parent.name == "web"

    def test_uploads_dir_with_env_var(self):
        """Test uploads directory with environment variable."""
        os.environ[UPLOADS_DIR] = "/tmp/test_uploads"
        result = get_uploads_dir()
        assert result == Path("/tmp/test_uploads")

    def test_uploads_dir_env_overrides_web_dir(self):
        """Test that UPLOADS_DIR env var overrides computed default."""
        os.environ[WEB_DIR] = "/custom/web"
        os.environ[UPLOADS_DIR] = "/custom/uploads"
        result = get_uploads_dir()
        assert result == Path("/custom/uploads")


class TestGetWebDir:
    """Test get_web_dir() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(WEB_DIR, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_web_dir(self):
        """Test default web directory path."""
        result = get_web_dir()
        assert result.name == "web"

    def test_web_dir_with_env_var(self):
        """Test web directory with environment variable."""
        os.environ[WEB_DIR] = "/custom/web"
        result = get_web_dir()
        assert result == Path("/custom/web")


class TestGetWebStaticDir:
    """Test get_web_static_dir() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(WEB_STATIC_DIR, None)
        os.environ.pop(WEB_DIR, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_static_dir(self):
        """Test default static directory path."""
        result = get_web_static_dir()
        assert result.name == "static"

    def test_static_dir_with_env_var(self):
        """Test static directory with environment variable."""
        os.environ[WEB_STATIC_DIR] = "/custom/static"
        result = get_web_static_dir()
        assert result == Path("/custom/static")


class TestGetDefaultUploadsDir:
    """Test get_default_uploads_dir() alias function."""

    def test_alias_returns_same_as_get_uploads_dir(self):
        """Test that alias function returns same result as get_uploads_dir()."""
        result1 = get_default_uploads_dir()
        result2 = get_uploads_dir()
        assert result1 == result2


class TestGetExternalUploadDirs:
    """Test get_external_upload_dirs() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(EXTERNAL_UPLOAD_DIRS, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_no_env_var_returns_empty_list(self):
        """Test that missing env var returns empty list."""
        result = get_external_upload_dirs()
        assert result == []

    def test_empty_env_var_returns_empty_list(self):
        """Test that empty env var returns empty list."""
        os.environ[EXTERNAL_UPLOAD_DIRS] = ""
        result = get_external_upload_dirs()
        assert result == []

    def test_nonexistent_dirs_are_filtered(self):
        """Test that nonexistent directories are not included."""
        os.environ[EXTERNAL_UPLOAD_DIRS] = "/nonexistent/path1,/nonexistent/path2"
        result = get_external_upload_dirs()
        assert result == []

    def test_existing_dirs_are_included(self):
        """Test that existing directories are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = Path(tmpdir) / "uploads1"
            dir2 = Path(tmpdir) / "uploads2"
            dir1.mkdir()
            dir2.mkdir()

            os.environ[EXTERNAL_UPLOAD_DIRS] = f"{dir1},{dir2}"
            result = get_external_upload_dirs()
            assert len(result) == 2
            assert dir1 in result
            assert dir2 in result


class TestGetExternalSkillsDirs:
    """Test get_external_skills_dirs() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(EXTERNAL_SKILLS_LIBRARY_DIRS, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_no_env_var_returns_empty_list(self):
        """Test that missing env var returns empty list."""
        result = get_external_skills_dirs()
        assert result == []

    def test_tilde_expansion(self):
        """Test that tilde (~) is expanded to home directory."""
        os.environ[EXTERNAL_SKILLS_LIBRARY_DIRS] = "~/skills"
        result = get_external_skills_dirs()
        assert len(result) == 1
        assert result[0] == Path.home() / "skills"

    def test_env_var_expansion(self):
        """Test that environment variables in paths are expanded."""
        os.environ["CUSTOM_SKILLS_DIR"] = "/opt/skills"
        os.environ[EXTERNAL_SKILLS_LIBRARY_DIRS] = "$CUSTOM_SKILLS_DIR"
        result = get_external_skills_dirs()
        assert len(result) == 1
        assert result[0] == Path("/opt/skills")

    def test_url_like_paths_are_skipped(self):
        """Test that URL-like paths are skipped with warning."""
        os.environ[EXTERNAL_SKILLS_LIBRARY_DIRS] = "https://example.com/skills"
        result = get_external_skills_dirs()
        assert result == []


class TestGetStorageRoot:
    """Test get_storage_root() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(STORAGE_ROOT, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_storage_root(self):
        """Test default storage root path."""
        result = get_storage_root()
        assert result == Path.home() / ".xagent"

    def test_storage_root_with_env_var(self):
        """Test storage root with environment variable."""
        os.environ[STORAGE_ROOT] = "/custom/storage"
        result = get_storage_root()
        assert result == Path("/custom/storage")


class TestGetSandboxImage:
    """Test get_sandbox_image() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(SANDBOX_IMAGE, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_sandbox_image(self):
        """Test default sandbox image name."""
        result = get_sandbox_image()
        assert result == "xprobe/xagent-sandbox:latest"

    def test_sandbox_image_with_env_var(self):
        """Test sandbox image with environment variable."""
        os.environ[SANDBOX_IMAGE] = "custom/sandbox:v1.0"
        result = get_sandbox_image()
        assert result == "custom/sandbox:v1.0"


class TestGetLancedbPath:
    """Test get_lancedb_path() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(LANCEDB_PATH, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_lancedb_path(self):
        """Test default LanceDB path (relative to cwd)."""
        result = get_lancedb_path()
        assert result == Path("data/lancedb")

    def test_lancedb_path_with_env_var(self):
        """Test LanceDB path with environment variable."""
        os.environ[LANCEDB_PATH] = "/custom/lancedb"
        result = get_lancedb_path()
        assert result == Path("/custom/lancedb")


class TestGetDefaultSqliteDbPath:
    """Test get_default_sqlite_db_path() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(STORAGE_ROOT, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_sqlite_db_path(self):
        """Test default SQLite database path."""
        result = get_default_sqlite_db_path()
        assert result == Path.home() / ".xagent" / "xagent.db"

    def test_sqlite_db_path_respects_storage_root(self):
        """Test that SQLite path respects STORAGE_ROOT env var."""
        os.environ[STORAGE_ROOT] = "/custom/storage"
        result = get_default_sqlite_db_path()
        assert result == Path("/custom/storage/xagent.db")


class TestGetDatabaseUrl:
    """Test get_database_url() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(DATABASE_URL, None)
        os.environ.pop(STORAGE_ROOT, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_database_url(self):
        """Test default database URL (SQLite)."""
        result = get_database_url()
        assert result.startswith("sqlite:///")
        assert result.endswith("xagent.db")

    def test_database_url_with_env_var(self):
        """Test database URL with environment variable."""
        os.environ[DATABASE_URL] = "postgresql://user:pass@localhost/db"
        result = get_database_url()
        assert result == "postgresql://user:pass@localhost/db"


class TestGetSandboxCpus:
    """Test get_sandbox_cpus() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(SANDBOX_CPUS, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_no_env_var_returns_none(self):
        """Test that missing env var returns None."""
        result = get_sandbox_cpus()
        assert result is None

    def test_valid_cpu_count(self):
        """Test valid CPU count from env var."""
        os.environ[SANDBOX_CPUS] = "4"
        result = get_sandbox_cpus()
        assert result == 4

    def test_invalid_cpu_count_returns_none(self):
        """Test that invalid CPU count returns None."""
        os.environ[SANDBOX_CPUS] = "invalid"
        result = get_sandbox_cpus()
        assert result is None


class TestGetSandboxMemory:
    """Test get_sandbox_memory() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(SANDBOX_MEMORY, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_no_env_var_returns_none(self):
        """Test that missing env var returns None."""
        result = get_sandbox_memory()
        assert result is None

    def test_valid_memory_value(self):
        """Test valid memory value from env var."""
        os.environ[SANDBOX_MEMORY] = "2048"
        result = get_sandbox_memory()
        assert result == 2048

    def test_invalid_memory_value_returns_none(self):
        """Test that invalid memory value returns None."""
        os.environ[SANDBOX_MEMORY] = "invalid"
        result = get_sandbox_memory()
        assert result is None


class TestGetSandboxEnv:
    """Test get_sandbox_env() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(SANDBOX_ENV, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_no_env_var_returns_empty_dict(self):
        """Test that missing env var returns empty dict."""
        result = get_sandbox_env()
        assert result == {}

    def test_empty_env_var_returns_empty_dict(self):
        """Test that empty env var returns empty dict."""
        os.environ[SANDBOX_ENV] = ""
        result = get_sandbox_env()
        assert result == {}

    def test_valid_env_config(self):
        """Test valid environment variable configuration."""
        os.environ[SANDBOX_ENV] = "KEY1=value1;KEY2=value2"
        result = get_sandbox_env()
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_env_config_with_spaces(self):
        """Test that spaces around keys/values are trimmed."""
        os.environ[SANDBOX_ENV] = " KEY1 = value1 ; KEY2 = value2 "
        result = get_sandbox_env()
        assert result == {"KEY1": "value1", "KEY2": "value2"}


class TestGetSandboxVolumes:
    """Test get_sandbox_volumes() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(SANDBOX_VOLUMES, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_no_env_var_returns_empty_list(self):
        """Test that missing env var returns empty list."""
        result = get_sandbox_volumes()
        assert result == []

    def test_empty_env_var_returns_empty_list(self):
        """Test that empty env var returns empty list."""
        os.environ[SANDBOX_VOLUMES] = ""
        result = get_sandbox_volumes()
        assert result == []

    def test_valid_volume_config(self):
        """Test valid volume configuration."""
        os.environ[SANDBOX_VOLUMES] = "/host:/container:ro"
        result = get_sandbox_volumes()
        assert len(result) == 1
        assert result[0] == ("/host", "/container", "ro")

    def test_volume_with_explicit_mode(self):
        """Test volume configuration with explicit mode."""
        os.environ[SANDBOX_VOLUMES] = "/host:/container:rw"
        result = get_sandbox_volumes()
        assert result[0][2] == "rw"

    def test_volume_defaults_to_readonly(self):
        """Test that volume defaults to readonly mode."""
        os.environ[SANDBOX_VOLUMES] = "/host:/container"
        result = get_sandbox_volumes()
        assert result[0][2] == "ro"

    def test_invalid_mode_defaults_to_readonly(self):
        """Test that invalid mode defaults to readonly."""
        os.environ[SANDBOX_VOLUMES] = "/host:/container:invalid"
        result = get_sandbox_volumes()
        assert result[0][2] == "ro"

    def test_tilde_expansion_in_volume_src(self):
        """Test that tilde is expanded in volume source path."""
        os.environ[SANDBOX_VOLUMES] = "~/data:/container:ro"
        result = get_sandbox_volumes()
        assert result[0][0] == str(Path.home() / "data")

    def test_multiple_volumes(self):
        """Test multiple volume configurations."""
        os.environ[SANDBOX_VOLUMES] = "/host1:/container1:ro;/host2:/container2:rw"
        result = get_sandbox_volumes()
        assert len(result) == 2
        assert result[0] == ("/host1", "/container1", "ro")
        assert result[1] == ("/host2", "/container2", "rw")


class TestGetBoxliteHomeDir:
    """Test get_boxlite_home_dir() function."""

    def setup_method(self):
        """Save and clear environment variables before each test."""
        self.original_env = os.environ.copy()
        os.environ.pop(BOXLITE_HOME_DIR, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_no_env_var_returns_none(self):
        """Test that missing env var returns None."""
        result = get_boxlite_home_dir()
        assert result is None

    def test_boxlite_home_dir_with_env_var(self):
        """Test BoxLite home directory with environment variable."""
        os.environ[BOXLITE_HOME_DIR] = "/custom/boxlite"
        result = get_boxlite_home_dir()
        assert result == Path("/custom/boxlite")
