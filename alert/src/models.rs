use chrono::NaiveDateTime;
use chrono::offset::Utc;
use crate::schema::{dollar, alerts};

#[derive(Queryable, Insertable, Debug)]
#[table_name="dollar"]
pub struct Dollar {
    pub id: Option<i32>,
    pub buy: f64,
    pub buy_amount: i32,
    pub sell: f64,
    pub sell_amount: i32,
    pub last: f64,
    pub var: f64,
    pub varper: f64,
    pub volume: i32,
    pub adjustment: f64,
    pub min: f64,
    pub max: f64,
    pub oin: i32,
    pub created_at: NaiveDateTime
}

impl Dollar {
    pub fn new() -> Dollar {
        let created_at = Utc::now().naive_utc();

        Dollar {
            id: None,
            buy: 0.0,
            buy_amount: 0,
            sell: 0.0,
            sell_amount: 0,
            last: 0.0,
            var: 0.0,
            varper: 0.0,
            volume: 0,
            adjustment: 0.0,
            min: 0.0,
            max: 0.0,
            oin: 0,
            created_at: created_at
        }
    }
}

#[derive(Queryable, Insertable, Debug)]
#[table_name="alerts"]
pub struct Alert {
    pub id: Option<i32>,
    pub created_at: NaiveDateTime,
    pub asset: String,
    pub previous_value: f64,
    pub current_value: f64,
    pub active: bool
}

impl Alert {
    pub fn new(asset: String, previous_value: f64, current_value: f64) -> Alert {
        let created_at = Utc::now().naive_utc();

        Alert {
            id: None,
            created_at: created_at,
            asset: asset,
            previous_value: previous_value,
            current_value: current_value,
            active: true
        }
    }
}
