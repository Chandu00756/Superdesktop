#!/usr/bin/env python3
"""Omega unified lightweight migration runner.
Supports SQLite now; structure allows extension to Postgres.
Stores applied migrations in table: schema_migrations(version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT).
"""
import os, sqlite3, glob, time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
TARGETS = {
    'backend': os.path.join(ROOT, 'backend', 'omega_control.db'),
    'orchestrator': os.path.join(ROOT, 'omega_orchestrator.db'),
}
MIG_DIRS = {
    'backend': os.path.join(ROOT, 'backend', 'migrations'),
    'orchestrator': os.path.join(ROOT, 'omega-orchestrator', 'migrations'),
}

def ensure_table(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)')
    conn.commit()

def applied_versions(conn):
    cur = conn.execute('SELECT version FROM schema_migrations')
    return {r[0] for r in cur.fetchall()}

def parse_migration_name(path):
    base = os.path.basename(path)
    if not base.lower().startswith('v'):
        return None
    parts = base.split('__',1)
    try:
        ver = int(parts[0][1:])
    except Exception:
        return None
    return ver, base

def apply_sql(conn, path, ver, name):
    with open(path,'r') as f:
        sql = f.read()
    try:
        conn.executescript(sql)
        conn.execute('INSERT INTO schema_migrations(version,name,applied_at) VALUES(?,?,?)', (ver, name, datetime.utcnow().isoformat()))
        conn.commit()
        print(f"Applied {name}")
    except Exception as e:
        print(f"Failed migration {name}: {e}")
        raise

def migrate(target: str):
    db_path = TARGETS[target]
    mig_dir = MIG_DIRS[target]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        ensure_table(conn)
        done = applied_versions(conn)
        files = sorted(glob.glob(os.path.join(mig_dir, 'V*.sql')))
        to_apply = []
        for f in files:
            parsed = parse_migration_name(f)
            if not parsed:
                continue
            ver, name = parsed
            if ver not in done:
                to_apply.append((ver, name, f))
        for ver,name,f in sorted(to_apply, key=lambda x:x[0]):
            apply_sql(conn, f, ver, name)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=TARGETS.keys(), required=True)
    args = ap.parse_args()
    migrate(args.target)
