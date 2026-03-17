//! Native test runner for json_prover logic
//! Run with: cargo run --bin test_native

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Transaction {
    pub from: String,
    pub to: String,
    pub amount: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Account {
    pub account_name: String,
    pub balance: u32,
}

/// Same logic as the WASM json_prover function
fn json_prover(data_str: &str, key_path: &str, account_json: &str, transactions_json: &str) -> String {
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

fn main() {
    // Load fixtures
    let data_str = include_str!("../../guest-programs/json-query/fixtures/test_input.json");
    let account_json = include_str!("../../guest-programs/json-query/fixtures/account.json");
    let transactions_json = include_str!("../../guest-programs/json-query/fixtures/transactions.json");

    println!("=== JSON Prover Native Test ===\n");
    println!("Input JSON size: {} bytes", data_str.len());
    println!("Account: {}", account_json.trim());
    println!("Transactions: {}", transactions_json.trim());

    // Test various query paths (3 levels of nesting)
    let test_queries = [
        "company.departments.engineering.teams.frontend.lead",
        "company.departments.engineering.teams.backend.members",
        "company.departments.sales.regions.south.achieved",
        "company.departments.hr.policies.remote_work.days_per_week",
        "company.metadata.version",
    ];

    println!("\n=== Query Results ===\n");

    for query in &test_queries {
        let result = json_prover(data_str, query, account_json, transactions_json);
        let parsed: Value = serde_json::from_str(&result).unwrap();

        println!("Query: {}", query);
        println!("  Extracted: {}", parsed["extracted_value"]);
        println!("  Final balance: {}", parsed["final_account"]["balance"]);
        println!();
    }

    // Verify expected results
    println!("=== Verification ===\n");

    let result = json_prover(
        data_str,
        "company.departments.engineering.teams.frontend.lead",
        account_json,
        transactions_json
    );
    let parsed: Value = serde_json::from_str(&result).unwrap();

    // Expected: alice starts with 1000, sends 100 to bob, receives 50 from bob, sends 200 to carol
    // Final: 1000 - 100 + 50 - 200 = 750
    let expected_balance = 750u32;
    let actual_balance = parsed["final_account"]["balance"].as_u64().unwrap() as u32;

    assert_eq!(actual_balance, expected_balance, "Balance mismatch!");
    println!("Balance check: {} == {} PASS", actual_balance, expected_balance);

    let expected_value = "Bob Smith";
    let actual_value = parsed["extracted_value"].as_str().unwrap();
    assert_eq!(actual_value, expected_value, "Extracted value mismatch!");
    println!("Value check: {:?} == {:?} PASS", actual_value, expected_value);

    println!("\nAll tests passed!");
}
