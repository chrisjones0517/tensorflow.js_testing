const tf = require('@tensorflow/tfjs');
const iris = require('./iris.json');
const irisTesting = require('./iris-testing');
require('@tensorflow/tfjs-node');  

// convert/setup our data
const trainingData = tf.tensor2d([iris.map(item => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
])]);

const outputData = tf.tensor2d([iris.map(item => [
  item.species === 'Iris-setosa' ? 1 : 0,
  item.species === 'Iris-virginica' ? 1 : 0,
  item.species === 'Iris-versicolor' ? 1 : 0
])]);

const testingData = tf.tensor2d([irisTesting.map(item => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
])]);

const model = tf.sequential();

model.add(tf.layers.dense({
  inputShape: [4],
  activation: "sigmoid",
  units: 5
}));

model.add(tf.layers.dense({
  inputShape: [5],
  activation: "sigmoid",
  units: 3
}));

model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 3
}));

model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.06)
})

const startTime = Date.now();
model.fit(trainingData, outputData, {epochs: 100})
  .then((history) => {
    console.log('Done!', Date.now() - startTime);
  })
// build neural network
// train/fit our network
// test network




