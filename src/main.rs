use ndarray::prelude::*;
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

trait Layer {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, out_grad: Array2<f64>, learning_rate: f64) -> Array2<f64>;
}

#[allow(dead_code)]
struct Dense {
    input_size: usize,
    output_size: usize,
    input: Array2<f64>,
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl Dense {
    fn new(input_size: usize, output_size: usize) -> Self {
        let input = Array2::zeros((0, input_size));
        let weights = Array2::random((input_size, output_size), StandardNormal);
        let bias = Array1::random((output_size,), StandardNormal);
        
        Dense {input_size, output_size, input, weights, bias}
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.input = input;
        self.input.dot(&self.weights) + &self.bias
    }

    fn backward(&mut self, out_grad: Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let mut input_grad = out_grad.dot(&self.weights.t());
        let mut weights_grad = self.input.t().dot(&out_grad);
        let bias_grad = out_grad.sum_axis(Axis(0));

        // gradient clipping
        let threshold = 1.0;
        input_grad.mapv_inplace(|x| {
            if x > threshold {threshold} 
            else if x < -threshold {-threshold}
            else {x}
        });

        weights_grad.mapv_inplace(|x| {
            if x > threshold {threshold} 
            else if x < -threshold {-threshold}
            else {x}
        });

        self.weights.zip_mut_with(&weights_grad, |w, g| *w -= g * learning_rate);
        self.bias.zip_mut_with(&bias_grad, |b, g| *b -= g * learning_rate);

        input_grad
    }
}

struct ReLu {
    input: Array2<f64>,
}

impl ReLu {
    fn new() -> Self {
        ReLu {input: Array2::zeros((0, 0))}
    }
}

impl Layer for ReLu {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.input = input;
        self.input.map(|x| x.max(0.0))
    }

    fn backward(&mut self, out_grad: Array2<f64>, _learning_rate: f64) -> Array2<f64> {
        let input_grad = self.input.map(|x| if x > &0.0 { 1.0 } else { 0.0 });
        input_grad * out_grad
    }
}

fn mean_squared_error(y_pred: &Array2<f64>, y_true: &Array2<f64>) -> Array2<f64> {
    0.5 * (y_pred - y_true).mapv_into(|x| x * x)
}

fn mean_squared_error_derivative(y_pred: &Array2<f64>, y_true: &Array2<f64>) -> Array2<f64> {
    -(y_pred - y_true)
}

struct Model {
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Model { layers }
    }

    fn predict(&mut self, x: Array2<f64>) -> Array2<f64> {
        let mut out = x;
        for layer in &mut self.layers {
            out = layer.forward(out);
        }
        out
    }

    fn fit(&mut self, x: Array2<f64>, y: Array2<f64>, epochs: u32, learning_rate: f64) {
        for e in 0..epochs {
            let y_pred = self.predict(x.clone()); // Clone x if necessary
            let loss = mean_squared_error(&y_pred, &y); // Clone y if necessary

            let loss_derivative = mean_squared_error_derivative(&y_pred, &y);

            let mut out_grad = loss_derivative;
            for layer in self.layers.iter_mut().rev() {
                out_grad = layer.backward(out_grad, learning_rate);
            }

            println!("{:?}/{:?} Loss: {:?}", e, epochs, loss);
        }
    }
}

fn main() {
    let x_values = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y_values = vec![[0.0], [1.0], [1.0], [0.0]];

    let x = Array2::from_shape_vec((4, 2),x_values.iter().flat_map(|row| row.iter()).cloned().collect(),).unwrap();
    let y = Array2::from_shape_vec((4, 1),y_values.iter().flat_map(|row| row.iter()).cloned().collect(),).unwrap();

    let mut model = Model::new(vec![
        Box::new(Dense::new(2, 8)),
        Box::new(ReLu::new()),
        Box::new(Dense::new(8, 4)),
        Box::new(ReLu::new()),
        Box::new(Dense::new(4, 1)),
    ]);

    model.fit(x, y, 100, 0.3)
}
