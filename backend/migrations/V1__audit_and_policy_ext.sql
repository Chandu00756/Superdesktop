-- V1 initial migration for backend advanced auditing & policy metadata
BEGIN;
CREATE TABLE IF NOT EXISTS audit_event_types (
    code TEXT PRIMARY KEY,
    description TEXT,
    severity_default TEXT,
    retention_days INTEGER DEFAULT 30
);
INSERT OR IGNORE INTO audit_event_types(code,description,severity_default,retention_days) VALUES
 ('node_join','Node join request received','info',30),
 ('node_approve','Node approved','info',30),
 ('node_deny','Node denied','warning',30),
 ('policy_violation','Policy violation detected','warning',90),
 ('security_alert','Security alert','critical',180),
 ('key_rotate','Key rotation executed','info',365),
 ('key_revoke','Key revoked','warning',365),
 ('session_start','Session started','info',30),
 ('session_end','Session ended','info',30),
 ('autoscale_decision','Autoscale decision taken','info',30);
CREATE TABLE IF NOT EXISTS audit_export_manifests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bundle_id TEXT UNIQUE,
    created_at REAL,
    format TEXT,
    record_count INTEGER,
    sha256 TEXT,
    signature TEXT
);
COMMIT;
