# Backend DB Migrations

This directory will contain versioned migration manifests for both SQLite (dev) and Postgres (future) schemas.

Format (per migration file name):
V{integer}__{slug}.sql

A companion JSON metadata may be added for reversible operations.
