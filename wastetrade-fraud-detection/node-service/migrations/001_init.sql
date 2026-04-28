-- =====================================================================
-- WasteTrade fraud detection schema.
-- Designed for MySQL 8+. Uses JSON columns for embeddings and pairwise
-- matrices - no vector database needed at this volume.
-- =====================================================================

CREATE TABLE IF NOT EXISTS listings (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  material_type   VARCHAR(255) NOT NULL,
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_listings_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS listing_images (
  id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  listing_id        BIGINT UNSIGNED NOT NULL,
  path              VARCHAR(1024) NOT NULL,
  filename          VARCHAR(512) NOT NULL,
  -- Embedding stored as JSON array of floats. NULL until first fraud check
  -- triggers embedding. Indexed only on the FK; we never query by embedding
  -- contents at this scale.
  embedding         JSON NULL,
  embedding_model   VARCHAR(64) NULL,
  created_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_listing_images_listing (listing_id),
  CONSTRAINT fk_listing_images_listing FOREIGN KEY (listing_id) REFERENCES listings(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS loads (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  listing_id      BIGINT UNSIGNED NOT NULL,
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_loads_listing (listing_id),
  CONSTRAINT fk_loads_listing FOREIGN KEY (listing_id) REFERENCES listings(id) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS load_images (
  id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  load_id           BIGINT UNSIGNED NOT NULL,
  path              VARCHAR(1024) NOT NULL,
  filename          VARCHAR(512) NOT NULL,
  embedding         JSON NULL,
  embedding_model   VARCHAR(64) NULL,
  created_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_load_images_load (load_id),
  CONSTRAINT fk_load_images_load FOREIGN KEY (load_id) REFERENCES loads(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- The big one: every fraud check is fully recorded so it can be replayed
-- against new thresholds without re-spending API money.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fraud_checks (
  id                       BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  listing_id               BIGINT UNSIGNED NOT NULL,
  load_id                  BIGINT UNSIGNED NOT NULL,
  claimed_material         VARCHAR(255) NULL,

  -- Raw embedding-derived signals.
  pairwise_scores          JSON NOT NULL,        -- [{listing_image_id, load_image_id, cosine}, ...]
  max_per_load_image       JSON NOT NULL,        -- per-load best match - what minOfMax was computed from
  aggregate_score          DECIMAL(6,4) NOT NULL,
  min_of_max               DECIMAL(6,4) NOT NULL,
  embedding_model          VARCHAR(64) NOT NULL,

  -- VLM signals.
  vlm_provider             VARCHAR(32) NOT NULL,
  vlm_model                VARCHAR(64) NOT NULL,
  vlm_verdict              ENUM('match','mismatch','uncertain') NOT NULL,
  vlm_confidence           DECIMAL(4,3) NOT NULL DEFAULT 0,
  vlm_discrepancies        JSON NULL,
  vlm_summary              TEXT NULL,
  vlm_set_a_description    TEXT NULL,
  vlm_set_b_description    TEXT NULL,
  vlm_comparison           TEXT NULL,

  -- Final decision and the rule that produced it.
  decision                 ENUM('pass','review','suspicious') NOT NULL,
  decision_reasons         JSON NOT NULL,
  thresholds_used          JSON NOT NULL,

  -- Reviewer feedback - filled in later via /fraud-check/:id/outcome.
  reviewer_outcome         ENUM('confirmed_fraud','false_alarm','inconclusive') NULL,
  reviewer_notes           TEXT NULL,
  reviewed_at              DATETIME NULL,

  elapsed_ms               INT UNSIGNED NOT NULL DEFAULT 0,
  created_at               DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (id),
  KEY idx_fraud_checks_listing (listing_id),
  KEY idx_fraud_checks_load (load_id),
  KEY idx_fraud_checks_decision (decision),
  KEY idx_fraud_checks_outcome (reviewer_outcome),
  KEY idx_fraud_checks_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
