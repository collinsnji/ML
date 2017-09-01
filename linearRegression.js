var fs = require('fs');

// helper function
function csvToArray(data) {
    var rows = data.split("\n");
    return rows.map(function (row) {
        return row.split(",");
    });
}

// error function
function computeError(b, m, points) {
    var totalError = 0;
    for (var i = 0; i < points.length; i++) {
        var x = Number(points[i][0]);
        var y = Number(points[i][1]);
        totalError += Math.pow((y - (m * x + b)), 2);
    }
    return totalError / parseFloat(points.length);
}

// gradient descent
function gradientDescent(b_current, m_current, points, learningRate) {
    var b_gradient = 0;
    var m_gradient = 0;
    var N = parseFloat(points.length);
    for (var i = 0; i < points.length; i++) {
        var x = Number(points[i][0]);
        var y = Number(points[i][1]);
        b_gradient += (-(2 / N) * (y - ((m_current * x) + b_current)));
        m_gradient += (-(2 / N) * x * (y - ((m_current * x) + b_current)));
    }
    var new_b = b_current - (learningRate * b_gradient);
    var new_m = m_current - (learningRate * m_gradient);
    return [new_b, new_m];
}

function gradientDescentRunner(points, starting_b, starting_m, learningRate, iterations) {
    var b = starting_b;
    var m = starting_m;

    for (var i = 0; i < iterations; i++) {
        [b, m] = gradientDescent(b, m, points, learningRate);
    }
    return [b, m];
}

// init
function init() {
    var points = csvToArray(fs.readFileSync(`${__dirname}/dataFiles/linearRegression.csv`).toString());
    // hyper parameters
    var learningRate = 0.0001;
    // y = mx + b
    var initial_b = 0;
    var initial_m = 0;
    var iterations = 1000;
    [b, m] = gradientDescentRunner(points, initial_b, initial_m, learningRate, iterations);
    console.log(`Running...\n`);
    console.log(`Initial values: b = ${initial_b}, m = ${initial_m}`);
    console.log(`After ${iterations} iterations:\n b = ${b}, m = ${m}, error = ${computeError(b, m, points)}`);
}

init();
