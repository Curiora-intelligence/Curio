"""Persistent users, conversations, memories and agent audit records."""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("users", sa.Column("id", sa.String(128), primary_key=True),
                    sa.Column("memory_version", sa.Integer, nullable=False, server_default="0"),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("conversations", sa.Column("id", sa.String(36), primary_key=True),
                    sa.Column("user_id", sa.String(128), sa.ForeignKey("users.id"), nullable=False),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.create_table("messages", sa.Column("id", sa.String(36), primary_key=True),
                    sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id"), nullable=False),
                    sa.Column("role", sa.String(24), nullable=False), sa.Column("content", sa.Text, nullable=False),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])
    op.create_table("memories", sa.Column("id", sa.String(36), primary_key=True),
                    sa.Column("user_id", sa.String(128), sa.ForeignKey("users.id"), nullable=False),
                    sa.Column("kind", sa.String(32), nullable=False), sa.Column("key", sa.String(128), nullable=False),
                    sa.Column("value", sa.Text, nullable=False), sa.Column("confidence", sa.Float, nullable=False),
                    sa.Column("importance", sa.Float, nullable=False),
                    sa.Column("source_message_id", sa.String(36), sa.ForeignKey("messages.id")),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
                    sa.Column("embedding", sa.JSON, nullable=True))
    op.create_index("ix_memories_user_id", "memories", ["user_id"])
    op.create_index("uq_active_memory", "memories", ["user_id", "kind", "key"], unique=True,
                    postgresql_where=sa.text("active"), sqlite_where=sa.text("active"))
    op.create_table("agent_runs", sa.Column("id", sa.String(36), primary_key=True),
                    sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id"), nullable=False),
                    sa.Column("request_id", sa.String(128)), sa.Column("status", sa.String(32), nullable=False),
                    sa.Column("iterations", sa.Integer, nullable=False), sa.Column("answer", sa.Text),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
                    sa.UniqueConstraint("conversation_id", "request_id"))
    op.create_index("ix_agent_runs_conversation_id", "agent_runs", ["conversation_id"])
    op.create_table("tool_calls", sa.Column("id", sa.String(36), primary_key=True),
                    sa.Column("agent_run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=False),
                    sa.Column("name", sa.String(128), nullable=False), sa.Column("arguments", sa.JSON, nullable=False),
                    sa.Column("result", sa.JSON), sa.Column("status", sa.String(32), nullable=False),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_tool_calls_agent_run_id", "tool_calls", ["agent_run_id"])


def downgrade():
    for name in ("tool_calls", "agent_runs", "memories", "messages", "conversations", "users"):
        op.drop_table(name)
