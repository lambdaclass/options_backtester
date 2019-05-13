extern crate fantoccini;
extern crate futures;
extern crate select;
extern crate serde_json;
extern crate tokio;
extern crate webdriver;

use fantoccini::{Client, Locator};
use futures::future::Future;
use futures::sync::oneshot;
use select::document::Document;
use select::predicate::Class;
use serde_json::json;
use webdriver::capabilities::Capabilities;

use crate::models::Dollar;

pub fn scrape() -> Dollar {
    let html = fetch_site();

    let document = Document::from(html.as_str());

    let mut value = Dollar::new();

    for node in document.find(Class("PriceCell")) {
        let full_class =
            node.attr("class")
            .unwrap_or_else(|| panic!("Node without class attribute"));
        let class = full_class.split_whitespace().last();

        match class {
            Some("bsz") => value.buy_amount = parse_i32(node.text()),
            Some("bid") => value.buy = parse_f64(node.text()),
            Some("ask") => value.sell = parse_f64(node.text()),
            Some("asz") => value.sell_amount = parse_i32(node.text()),
            Some("lst") => value.last = parse_f64(node.text()),
            Some("variation") => value.var = parse_f64(node.text()),
            Some("change") => {
                value.varper = parse_f64(node.text().trim_end_matches("%").to_string())
            }
            Some("von") => value.volume = parse_i32(node.text()),
            Some("settlementPrice") => value.adjustment = parse_f64(node.text()),
            Some("low") => value.min = parse_f64(node.text()),
            Some("hgh") => value.max = parse_f64(node.text()),
            Some("oin") => value.oin = parse_i32(node.text()),
            Some("futureImpliedRate") => {}
            Some(class) => {
                panic!("Non-matching class: {}", class);
            }
            None => {
                panic!("Node without empty class attribute");
            }
        }
    }

    value
}

fn fetch_site() -> String {
    // TODO: Set headless capabilities for chromedriver.
    let mut cap = Capabilities::new();
    let arg = json!({"args": ["-headless"]});
    cap.insert("moz:firefoxOptions".to_string(), arg);
    let c = Client::with_capabilities("http://localhost:4444", cap);
    let (sender, receiver) = oneshot::channel::<String>();

    tokio::run(
        c.map_err(|e| unimplemented!("failed to connect to WebDriver: {:?}", e))
            .and_then(|c| c.goto("https://rofex.primary.ventures/rofex/futuros"))
            .and_then(|mut c| c.current_url().map(move |url| (c, url)))
            .and_then(|(c, _url)| {
                c.wait_for_find(Locator::XPath("((//div[@class='PricePanelRow-row'])[1]//div[@class='PriceCell lst'])[not(contains(., '-'))]/.."))
            })
            .and_then(|mut e| {
                e.html(true)
            })
            .and_then(|e| match sender.send(e) {
                Err(err) => panic!("Error sending fetched site: {}", err),
                Ok(()) => Ok(()),
            })
            .map_err(|e| {
                panic!("a WebDriver command failed: {:?}", e);
            }),
    );

    receiver.wait().unwrap()
}

fn parse_i32(text: String) -> i32 {
    text.replace(".", "")
        .trim()
        .parse()
        .unwrap_or_else(|err| panic!("Error parsing \"{}\": {}", text, err))
}

fn parse_f64(text: String) -> f64 {
    text.replace(".", "")
        .replace(",", ".")
        .trim()
        .parse()
        .unwrap_or_else(|err| panic!("Error parsing \"{}\": {}", text, err))
}
