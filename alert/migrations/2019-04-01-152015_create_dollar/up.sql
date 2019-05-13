CREATE TABLE dollar (
  id              INTEGER PRIMARY KEY,
  buy             DOUBLE NOT NULL,
  buy_amount      INTEGER NOT NULL,
  sell            DOUBLE NOT NULL,
  sell_amount     INTEGER NOT NULL,
  last            DOUBLE NOT NULL,
  var             DOUBLE NOT NULL,
  varper          DOUBLE NOT NULL,
  volume          INTEGER NOT NULL,
  adjustment      DOUBLE NOT NULL,
  min             DOUBLE NOT NULL,
  max             DOUBLE NOT NULL,
  oin             INTEGER NOT NULL,
  created_at      DATETIME NOT NULL
);
