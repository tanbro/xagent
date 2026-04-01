#!/usr/bin/env python3
"""
Migrate uploaded_files.storage_path from absolute to relative paths.

This script converts existing absolute path records to relative paths.
Only converts paths that match the current UPLOADS_DIR configuration,
so it can be run multiple times safely (e.g., after changing XAGENT_UPLOADS_DIR).

Usage:
    python scripts/migrate_uploads_file_abs_path.py [--dry-run]

Options:
    --dry-run    Show what would be changed without making changes
    --confirm    Require confirmation before proceeding
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f".env file not found at {env_path}")
except ImportError:
    logger.warning("python-dotenv not available, skipping .env loading")


def get_database_url():
    """Get database URL from environment or default."""
    db_path = os.environ.get("DATABASE_URL")
    if db_path:
        if "://" in db_path:
            return db_path
        if Path(db_path).exists():
            return f"sqlite:///{db_path}"

    default_db = Path(__file__).parent.parent / "xagent.db"
    if default_db.exists():
        return f"sqlite:///{default_db}"

    data_db = Path(__file__).parent.parent / "data" / "xagent.db"
    if data_db.exists():
        return f"sqlite:///{data_db}"

    raise FileNotFoundError(
        "Database file not found. Please set DATABASE_URL environment variable."
    )


def migrate_storage_paths(dry_run: bool = False, confirm: bool = False) -> None:
    """Migrate absolute paths to relative paths in uploaded_files table."""
    from xagent.web.config import UPLOADS_DIR
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.utils.file import to_relative_path

    uploads_dir = Path(UPLOADS_DIR).resolve()

    logger.info(f"UPLOADS_DIR: {uploads_dir}")
    logger.info("Scanning for absolute path records...")

    # Setup database connection
    try:
        db_url = get_database_url()
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Find all records with absolute paths
        absolute_path_records = []
        for record in db.query(UploadedFile).all():
            storage_path = Path(record.storage_path)  # pyright: ignore[reportArgumentType]
            if storage_path.is_absolute():
                absolute_path_records.append(record)

        total = len(absolute_path_records)
        if total == 0:
            logger.info("No absolute path records found. Nothing to migrate.")
            return

        logger.info(f"Found {total} records with absolute paths")

        # Categorize records
        migrated = []
        skipped = []
        errors = []

        for record in absolute_path_records:
            abs_path = Path(record.storage_path)
            user_id = record.user_id

            try:
                # Check if path matches current UPLOADS_DIR structure
                user_root = uploads_dir / f"user_{user_id}"
                relative = to_relative_path(abs_path, user_id)

                # Verify the converted path points to the same location
                reconstructed = user_root / relative
                if reconstructed.resolve() != abs_path.resolve():
                    skipped.append(
                        {
                            "file_id": record.file_id,
                            "storage_path": record.storage_path,
                            "reason": "Path doesn't match current UPLOADS_DIR structure",
                        }
                    )
                    continue

                migrated.append(
                    {
                        "file_id": record.file_id,
                        "old_path": record.storage_path,
                        "new_path": relative,
                        "user_id": user_id,
                    }
                )

            except ValueError as e:
                # Path outside UPLOADS_DIR - skip
                skipped.append(
                    {
                        "file_id": record.file_id,
                        "storage_path": record.storage_path,
                        "reason": f"Outside UPLOADS_DIR: {e}",
                    }
                )
            except Exception as e:
                errors.append(
                    {
                        "file_id": record.file_id,
                        "storage_path": record.storage_path,
                        "error": str(e),
                    }
                )

        # Summary
        logger.info("=" * 60)
        logger.info("Migration Summary:")
        logger.info(f"  Total absolute path records: {total}")
        logger.info(f"  Can be migrated:            {len(migrated)}")
        logger.info(f"  Skipped (no match):         {len(skipped)}")
        logger.info(f"  Errors:                     {len(errors)}")
        logger.info("=" * 60)

        if errors:
            logger.warning(f"Errors encountered ({len(errors)}):")
            for err in errors[:5]:
                logger.warning(f"  - {err['file_id']}: {err['error']}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors) - 5} more")

        if skipped:
            logger.info(f"Skipped records ({len(skipped)}):")
            for skip in skipped[:5]:
                logger.info(f"  - {skip['file_id']}: {skip['reason']}")
            if len(skipped) > 5:
                logger.info(f"  ... and {len(skipped) - 5} more")

        if not migrated:
            logger.info("No records to migrate.")
            return

        # Show migration details
        logger.info(f"\nRecords to migrate ({len(migrated)}):")
        for m in migrated[:10]:
            logger.info(f"  {m['file_id']}: {m['old_path']}")
            logger.info(f"    -> {m['new_path']}")
        if len(migrated) > 10:
            logger.info(f"  ... and {len(migrated) - 10} more")

        # Confirm if needed
        if confirm and not dry_run:
            response = input("\nProceed with migration? (yes/no): ")
            if response.lower() not in ("yes", "y"):
                logger.info("Migration cancelled.")
                return

        # Execute migration
        if dry_run:
            logger.info("\n🔍 DRY RUN MODE - No changes made")
        else:
            logger.info("\nMigrating...")
            for m in migrated:
                record = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.file_id == m["file_id"])
                    .first()
                )
                if record:
                    record.storage_path = m["new_path"]

            try:
                db.commit()
                logger.info(f"✅ Successfully migrated {len(migrated)} records")
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Migration failed: {e}")
                sys.exit(1)

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate uploaded_files.storage_path from absolute to relative paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to see what would be changed
    python scripts/migrate_storage_paths.py --dry-run

    # Migrate with confirmation prompt
    python scripts/migrate_storage_paths.py --confirm

    # Migrate directly (use with caution)
    python scripts/migrate_storage_paths.py
        """,
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without making them"
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Require confirmation before proceeding",
    )

    args = parser.parse_args()

    logger.info("Starting storage path migration...")
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No changes will be made")

    try:
        migrate_storage_paths(dry_run=args.dry_run, confirm=args.confirm)
        if not args.dry_run:
            logger.info("✅ Migration completed successfully")
    except KeyboardInterrupt:
        logger.info("\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
