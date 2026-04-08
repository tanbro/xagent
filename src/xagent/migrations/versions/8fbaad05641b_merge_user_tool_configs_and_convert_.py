"""merge user_tool_configs and convert_storage_path

Revision ID: 8fbaad05641b
Revises: 20260403_add_user_tool_configs, 3da89273f616
Create Date: 2026-04-08 22:06:35.637624

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "8fbaad05641b"
down_revision: Union[str, None] = ("20260403_add_user_tool_configs", "3da89273f616")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
