extern crate diesel;
extern crate dotenv;

use diesel::prelude::*;
use diesel::sqlite::SqliteConnection;
use diesel::result::Error::NotFound;
use chrono::{Utc, NaiveDateTime, NaiveTime};
use std::env;
use crate::models::{Dollar, Alert};
use crate::schema::{dollar, alerts};

pub fn establish_connection() -> SqliteConnection {
    // We can unwrap safely because env variables are checked in main.rs::init_env()
    let database_url = env::var("DATABASE_URL").unwrap();
    SqliteConnection::establish(&database_url)
        .expect(&format!("Error connecting to {}", database_url))
}

pub fn store(conn: &SqliteConnection, new_dollar: &Dollar) {
    diesel::insert_into(dollar::table)
        .values(new_dollar)
        .execute(conn)
        .expect("Error saving new dollar");
}

pub fn get_dollar_on_close(conn: &SqliteConnection) -> Option<Dollar> {
    let yesterday = Utc::today().naive_utc().pred();
    let timestamp = NaiveDateTime::new(yesterday, NaiveTime::from_hms(23, 59, 59));

    let result = dollar::table.filter(dollar::created_at.lt(timestamp))
        .order(dollar::created_at.desc())
        .first::<Dollar>(conn);

    match result {
        Ok(previous) => Some(previous),
        Err(NotFound) => None,
        Err(err) => panic!("Error getting previous dollar: \n{}", err)
    }
}

pub fn store_alert(conn: &SqliteConnection, new_alert: &Alert) {
    diesel::insert_into(alerts::table)
        .values(new_alert)
        .execute(conn)
        .expect("Error saving new alert");
}

pub fn get_dollar_alert(conn: &SqliteConnection) -> Option<Alert> {
    let result = alerts::table.filter(alerts::asset.eq("dollar").and(alerts::active.eq(true)))
        .first::<Alert>(conn);

    match result {
        Ok(alert) => Some(alert),
        Err(NotFound) => None,
        Err(err) => panic!("Error getting current alert: \n{}", err)
    }
}

pub fn deactivate_alert(conn: &SqliteConnection, alert: &Alert) {
    diesel::update(alerts::table.filter(alerts::id.eq(alert.id)))
        .set(alerts::active.eq(false))
        .execute(conn)
        .expect("Error deactivating alert");
}

pub fn replace_alert(conn: &SqliteConnection, old_alert: &Alert, new_alert: &Alert) {
    // TODO: Do this operations inside a transaction for atomicity
    store_alert(conn, new_alert);
    deactivate_alert(conn, old_alert);
}
