"""Initial schema for scan_events, text_annotations, charging_stations, evse_assets

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-08-24 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Charging Stations table
    op.create_table(
        'charging_stations',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('operator_id', sa.String(length=10), nullable=False, index=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False, index=True),
        sa.Column('longitude', sa.Float(), nullable=False, index=True),
        sa.Column('address', sa.String(length=255), nullable=True),
        sa.Column('country_code', sa.String(length=2), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    )

    # 2. EVSE Assets table
    op.create_table(
        'evse_assets',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('station_id', sa.String(length=64), sa.ForeignKey('charging_stations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('standard_type', sa.String(length=20), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    )

    # 3. Scan Events table
    op.create_table(
        'scan_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('timestamp_utc', sa.DateTime(), server_default=sa.text('now()'), nullable=False, index=True),
        sa.Column('latitude', sa.Float(), nullable=False, index=True),
        sa.Column('longitude', sa.Float(), nullable=False, index=True),
        sa.Column('location_accuracy_m', sa.Float(), nullable=True),
        sa.Column('environmental_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('s3_object_uri', sa.String(length=512), nullable=False, unique=True),
        sa.Column('is_annotated', sa.Boolean(), server_default=sa.text('false'), nullable=False, index=True),
        sa.Column('training_iteration', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('is_valid_format', sa.Boolean(), nullable=True),
    )

    # 4. Text Annotations table
    op.create_table(
        'text_annotations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('scan_event_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('scan_events.id', ondelete='CASCADE'), nullable=False),
        sa.Column('extracted_text', sa.String(length=255), nullable=True),
        sa.Column('provenance', sa.String(length=100), nullable=False, index=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    )

def downgrade() -> None:
    op.drop_table('text_annotations')
    op.drop_table('scan_events')
    op.drop_table('evse_assets')
    op.drop_table('charging_stations')
