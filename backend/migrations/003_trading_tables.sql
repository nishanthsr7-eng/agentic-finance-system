-- Marketplace paper trading (was trading_api.ensure_trading_schema).
CREATE TABLE IF NOT EXISTS trading_wallet (
    user_id    INT PRIMARY KEY,
    cash_usd   DECIMAL(18,2) NOT NULL DEFAULT 100000.00,
    seeded_usd DECIMAL(18,2) NOT NULL DEFAULT 100000.00,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_wallet_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS watchlist (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT NOT NULL,
    symbol     VARCHAR(40) NOT NULL,
    category   VARCHAR(20),
    name       VARCHAR(160),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_watch (user_id, symbol),
    CONSTRAINT fk_watch_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS market_alerts (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    user_id      INT NOT NULL,
    symbol       VARCHAR(40) NOT NULL,
    name         VARCHAR(160),
    direction    VARCHAR(8) NOT NULL DEFAULT 'above',
    target_price DECIMAL(20,8) NOT NULL,
    active       TINYINT(1) DEFAULT 1,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    triggered_at DATETIME NULL,
    CONSTRAINT fk_alert_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_alert_user (user_id, active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
