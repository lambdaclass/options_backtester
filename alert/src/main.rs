#[macro_use]
extern crate diesel;
extern crate dotenv;
extern crate chrono;

pub mod schema;
pub mod models;

mod scrape;
mod storage;
mod alert;

use dotenv::dotenv;
use std::{thread, time, env};

fn main() {
    init_env();

    loop {
        std::thread::spawn(|| {
            let dbconn = crate::storage::establish_connection();
            let value = crate::scrape::scrape();
            crate::alert::check_dollar(&dbconn, &value);
            crate::storage::store(&dbconn, &value);
        });

        thread::sleep(time::Duration::from_secs(60));
    }
}

fn init_env() {
    dotenv().ok();
    env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set");
    env::var("SLACK_WEBHOOK_URL")
        .expect("SLACK_WEBHOOK_URL must be set");
}
