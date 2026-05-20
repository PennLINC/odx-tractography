use odx_rs::OdxDataset;
use std::env;
use std::path::Path;
fn main() {
    let path = env::args().nth(1).expect("usage: list_dpv <odx>");
    let ds = OdxDataset::open(Path::new(&path)).expect("open");
    println!("DPV in {}:", path);
    for n in ds.dpv_names() {
        println!("  - {}", n);
    }
    println!("DPF in {}:", path);
    for n in ds.dpf_names() {
        println!("  - {}", n);
    }
}
