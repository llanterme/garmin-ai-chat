"""Add conversation tables for chat functionality

Revision ID: b7d2e8a4f9c6
Revises: ace1f49c61c5
Create Date: 2025-08-27 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'b7d2e8a4f9c6'
down_revision = 'ace1f49c61c5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add conversation tables for persistent chat storage."""
    # Create conversations table
    op.create_table('conversations',
        sa.Column('id', mysql.CHAR(36), nullable=False),
        sa.Column('user_id', mysql.CHAR(36), nullable=False),
        sa.Column('title', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.Index('ix_conversations_user_id', 'user_id'),
        sa.Index('ix_conversations_created_at', 'created_at'),
    )
    
    # Create conversation_messages table
    op.create_table('conversation_messages',
        sa.Column('id', mysql.CHAR(36), nullable=False),
        sa.Column('conversation_id', mysql.CHAR(36), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),  # 'user', 'assistant', 'system'
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.Index('ix_conversation_messages_conversation_id', 'conversation_id'),
        sa.Index('ix_conversation_messages_created_at', 'created_at'),
    )
    
    # Create vector_ingestion_status table to track embedding sync
    op.create_table('vector_ingestion_status',
        sa.Column('id', mysql.CHAR(36), nullable=False),
        sa.Column('user_id', mysql.CHAR(36), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),  # 'in_progress', 'completed', 'failed'
        sa.Column('total_activities', sa.Integer(), default=0, nullable=False),
        sa.Column('processed_activities', sa.Integer(), default=0, nullable=False),
        sa.Column('failed_activities', sa.Integer(), default=0, nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('batch_size', sa.Integer(), default=10, nullable=False),
        sa.Column('force_reingest', sa.Boolean(), default=False, nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.Index('ix_vector_ingestion_status_user_id', 'user_id'),
        sa.Index('ix_vector_ingestion_status_started_at', 'started_at'),
    )


def downgrade() -> None:
    """Remove conversation tables."""
    op.drop_table('vector_ingestion_status')
    op.drop_table('conversation_messages')
    op.drop_table('conversations')