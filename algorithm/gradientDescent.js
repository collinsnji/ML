// Simple Gradient Descent
// Applied on function: 2x^2 - x

var fx = x => 4 * x - 1;
var current = Math.floor(Math.random() * 10);
var iterations = 0;
var step = 0.1;
var learingRate = 0.00000001;
var previous = current;

console.log(`Staring Gradient Descent at: current = ${current}\nRunning...`);
while (previous > learingRate) {
    var prev = current;
    current += -step * fx(prev);
    previous = Math.abs(current - prev);
    iterations++;
}

console.log(`After ${iterations} iterations, current = ${current}`);