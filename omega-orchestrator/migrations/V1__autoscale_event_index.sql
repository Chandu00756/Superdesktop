-- Ensure table exists (for SQLite fresh DB) before creating index
CREATE TABLE IF NOT EXISTS autoscaling_events (
	id TEXT PRIMARY KEY,
	ts TEXT,
	action TEXT NOT NULL,
	reason TEXT,
	util_before REAL,
	active_nodes INTEGER
);
CREATE INDEX IF NOT EXISTS idx_autoscaling_events_ts ON autoscaling_events(ts);
