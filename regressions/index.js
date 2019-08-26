require('@tensorflow/tfjs-node');
const LinearRegression = require('./linear-regression');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight', 'cylinders'],
  labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
  learningRate: 1,
  iterations: 100
});

regression.train();

regression.test(testFeatures, testLabels);
