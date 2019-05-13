extern crate reqwest;
extern crate serde_json;

use diesel::sqlite::SqliteConnection;
use serde_json::json;
use std::env;
use crate::models::{Dollar, Alert};
use crate::storage;

pub fn check_dollar(conn: &SqliteConnection, dollar: &Dollar) {
    let dollar_close = storage::get_dollar_on_close(conn);
    let closing_prc_change =
        match &dollar_close {
            Some(dollar_close) => {
                (dollar.last - dollar_close.last)/dollar_close.last * 100.0
            },
            None => 0.0
        };

    match storage::get_dollar_alert(conn) {
        Some(dollar_alert) => {
            let alert_prc_change = (dollar.last - dollar_alert.current_value)/dollar_alert.current_value * 100.0;
            // TODO: This value should be set by arguments to binary or by config files
            if alert_prc_change.abs() > 2.0 {
                if closing_prc_change.abs() > 3.0 {
                    // If we couldn't unwrap closing_prc_change == 0.0
                    let dollar_close = dollar_close.unwrap();
                    let new_alert = Alert::new("dollar".to_string(), dollar_alert.previous_value, dollar.last);
                    storage::replace_alert(conn, &dollar_alert, &new_alert);
                    send_updated_alert(dollar_close.last, dollar_alert.current_value, dollar.last, alert_prc_change, closing_prc_change);
                } else {
                    storage::deactivate_alert(conn, &dollar_alert);
                    send_resolved_alert();
                }
            }
        }
        None => {
            // TODO: This value should be set by arguments to binary or by config files
            if closing_prc_change.abs() > 3.0 {
                // If we couldn't unwrap closing_prc_change == 0.0
                let dollar_close = dollar_close.unwrap();
                let alert = Alert::new(String::from("dollar"), dollar_close.last, dollar.last);
                storage::store_alert(conn, &alert);
                send_new_alert(dollar_close.last, dollar.last, closing_prc_change);
            }
        }
    };
}
// payload = {
//     "username": "Financial Watchdog",
//     "icon_emoji": ":money_with_wings:",
//     "attachments": [{
//         "color": color,
//         "title": title,
//         "text": text,
//         "fallback": text
//         "footer": "Financial Watchdog",
//     }]
// }
fn send_new_alert(price_close: f64, price_current: f64, prc_change: f64) {
    let msg = format!("Closing price: {:.2}\nCurrent price: {:.2}\nChanged: {:+.2}%", price_close, price_current, prc_change);
    send_alert_slack("#B22222", "Price jump from closing", &msg);
    // TODO: send_alert_mail(prc_change);
}

fn send_resolved_alert() {
    let msg = format!("Dollar price from closing price is back below threshold");
    send_alert_slack("#3873AD", "Price back to normal", &msg);
    // TODO: send_alert_mail(prc_change);
}

fn send_updated_alert(price_close: f64, price_previous: f64, price_current: f64, prc_change: f64, prc_change_close: f64) {
    let msg = format!("Closing price: {:.2}\nPrevious price: {:.2}\nCurrent price: {:.2}\nChange from previous: *{:+.2}%*\nChange from close: {:+.2}%", price_close, price_previous, price_current, prc_change, prc_change_close);
    send_alert_slack("#f1a031", "[UPDATE] Price jump", &msg);
    // TODO: send_alert_mail(prc_change);
}

fn send_alert_slack(color: &str, title: &str, msg: &str) {
    let payload = json!({
            "username": "Financial Watchdog",
            "icon_emoji": ":money_with_wings:",
            "attachments": [{
                "color": color,
                "title": title,
                "text": msg,
                "fallback": msg,
                "footer": "Financial Watchdog"
            }]
        })
        .to_string();
    // We can unwrap safely because env variables are checked in main.rs::init_env()
    let slack_webhook_url = env::var("SLACK_WEBHOOK_URL").unwrap();
    let client = reqwest::Client::new();
    client.post(&slack_webhook_url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(payload)
        .send()
        .expect("Failed sending slack alert");
}
