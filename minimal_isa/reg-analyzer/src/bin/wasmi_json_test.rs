fn main() {
    let wasm_bytes = std::fs::read("../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm").unwrap();
    let engine = wasmi::Engine::default();
    let module = wasmi::Module::new(&engine, &wasm_bytes[..]).unwrap();
    let mut store = wasmi::Store::new(&engine, ());
    let linker = wasmi::Linker::<()>::new(&engine);
    let instance = linker.instantiate(&mut store, &module).unwrap().start(&mut store).unwrap();
    
    let memory = instance.get_memory(&store, "memory").unwrap();
    let parse_fn = instance.get_typed_func::<(i32, i32), i32>(&store, "parse_json_deep").unwrap();
    
    // Simple test first
    let simple = r#"{"a":1,"b":"hello"}"#;
    memory.write(&mut store, 0, simple.as_bytes()).unwrap();
    let r1 = parse_fn.call(&mut store, (0, simple.len() as i32)).unwrap();
    println!("Simple JSON ({} bytes): nodes = {}", simple.len(), r1);
    
    // Generate test JSON
    let mut json = String::from(r#"{"data":{"users":["#);
    let tpl = r#"{"name":"user_XXX","email":"user_XXX@example.com","age":25,"active":true,"tags":["a","b","c"]}"#;
    let mut i = 0;
    while json.len() < 2048 {
        if i > 0 { json.push(','); }
        json.push_str(&tpl.replace("XXX", &format!("{:04}", i)));
        i += 1;
    }
    json.push_str(r#"]},"meta":{"count":"#);
    json.push_str(&i.to_string());
    json.push_str(r#"}}"#);
    
    println!("Generated JSON: {} bytes", json.len());
    memory.write(&mut store, 0, json.as_bytes()).unwrap();
    let r2 = parse_fn.call(&mut store, (0, json.len() as i32)).unwrap();
    println!("Generated JSON: nodes = {}", r2);
}
