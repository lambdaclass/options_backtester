CREATE TABLE alerts (
  id                  INTEGER PRIMARY KEY,
  created_at          DATETIME NOT NULL,
  asset               VARCHAR(20) NOT NULL,
  previous_value      DOUBLE NOT NULL,
  current_value       DOUBLE NOT NULL,
  active              BOOLEAN NOT NULL
);
