**A Rust implementation of a Small Artificial Neural Network.**

I thought it would be very hard to work with Rust's ownership and borrowing mechanism for the flow of the data (from one layer to another), but it turned out to be pretty easy.

The network itself isn't that useful, it struggls to learn even the simplest data. It can be improved but that would be pointless. 

The network usage:

```Rust
main.rs

use ndarray::prelude::*;

fn main() {
    let x_values = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y_values = vec![[0.0], [1.0], [1.0], [0.0]];

    let x = Array2::from_shape_vec((4, 2),x_values.iter().flat_map(|row| row.iter()).cloned().collect()).unwrap();
    let y = Array2::from_shape_vec((4, 1),y_values.iter().flat_map(|row| row.iter()).cloned().collect()).unwrap();

    let mut model = Model::new(vec![
        Box::new(Dense::new(2, 8)),
        Box::new(ReLu::new()),
        Box::new(Dense::new(8, 4)),
        Box::new(ReLu::new()),
        Box::new(Dense::new(4, 1)),
    ]);

    model.fit(x, y, 100, 0.3)
}
```
Honestly, I created this to kill some time, no profits no loss, lol!