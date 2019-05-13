table! {
    alerts (id) {
        id -> Nullable<Integer>,
        created_at -> Timestamp,
        asset -> Text,
        previous_value -> Double,
        current_value -> Double,
        active -> Bool,
    }
}

table! {
    dollar (id) {
        id -> Nullable<Integer>,
        buy -> Double,
        buy_amount -> Integer,
        sell -> Double,
        sell_amount -> Integer,
        last -> Double,
        var -> Double,
        varper -> Double,
        volume -> Integer,
        adjustment -> Double,
        min -> Double,
        max -> Double,
        oin -> Integer,
        created_at -> Timestamp,
    }
}

allow_tables_to_appear_in_same_query!(
    alerts,
    dollar,
);
