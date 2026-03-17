//! JSON prover compiled to WASM for instruction analysis
//! Same logic as SP1's json example but without zkvm dependencies

use serde::{Deserialize, Serialize};
use serde_json::Value;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Transaction {
    pub from: String,
    pub to: String,
    pub amount: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Account {
    pub account_name: String,
    pub balance: u32,
}

/// Same logic as the SP1 JSON guest program
#[wasm_bindgen]
pub fn json_prover(data_str: &str, key_path: &str, account_json: &str, transactions_json: &str) -> String {
    // Parse inputs
    let mut account: Account = serde_json::from_str(account_json).unwrap();
    let transactions: Vec<Transaction> = serde_json::from_str(transactions_json).unwrap();
    
    // Parse JSON
    let v: Value = serde_json::from_str(data_str).unwrap();
    
    // Navigate nested path (e.g., "data.user.email" -> v["data"]["user"]["email"])
    let mut current = &v;
    for key in key_path.split('.') {
        current = &current[key];
    }
    let val = current.clone();
    
    // Process transactions (same as SP1 guest)
    for tx in transactions {
        if tx.from == account.account_name {
            account.balance -= tx.amount;
        }
        if tx.to == account.account_name {
            account.balance += tx.amount;
        }
    }
    
    // Return result as JSON
    serde_json::json!({
        "extracted_value": val,
        "final_account": account
    }).to_string()
}
