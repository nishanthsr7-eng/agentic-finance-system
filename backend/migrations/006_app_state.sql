-- One-row-per-key app state, e.g. demo_seeded_on for the rolling demo seed.
CREATE TABLE IF NOT EXISTS app_state (
    k          VARCHAR(64) PRIMARY KEY,
    v          VARCHAR(255),
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
