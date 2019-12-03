// CPPN step generator and render on canvas
// Author: Li Ding
// Last Update: Oct. 2018
// Source:
// http://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf
// http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/
// https://js.tensorflow.org/


// prevent memory overflow
'use strict';

var canvas, ctx, w, h, cppn, model, input, zstart, img;

// build cppn model
function build(h, w) {
  let x = tf.range(0, w, 1);
  x = x.tile([h]).reshape([h, w, 1]);
  x = x.div(w).mul(8).sub(4)
  let y = tf.range(0, h, 1);
  y = y.tile([w]).reshape([w, h, 1]).transpose([1, 0, 2]);
  y = y.div(h).mul(2).sub(1)
  const r = tf.sqrt(x.pow(2).add(y.pow(2)));
  const input = tf.concat([x, y, r], -1).reshape([1, h, w, 3]);
  const model_input = tf.input({ shape: [h, w, 11] })
  const conv1 = tf.layers.conv2d({
    inputShape: [h, w, 11],
    kernelSize: 1,
    filters: 32,
    strides: 1,
    activation: 'tanh',
    padding: 'same',
    kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 2 }),
    useBias: false
  }).apply(model_input);
  const conv2 = tf.layers.conv2d({
    kernelSize: 1,
    filters: 32,
    strides: 1,
    activation: 'tanh',
    padding: 'same',
    kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 2 }),
    useBias: false
  }).apply(conv1);
  const conv3 = tf.layers.conv2d({
    kernelSize: 1,
    filters: 32,
    strides: 1,
    activation: 'tanh',
    padding: 'same',
    kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 2 }),
    useBias: false
  }).apply(conv2);
  const conv5 = tf.layers.conv2d({
    kernelSize: 1,
    filters: 1,
    strides: 1,
    padding: 'same',
    kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 2 }),
    useBias: false
  }).apply(conv3);
  //const output = tf.layers.activation({activation:'sigmoid'}).apply(sub);
  let model = tf.model({ inputs: model_input, outputs: conv5 });
  return [input, model];
};


// on load function
function init() {
  // make canvas to fit window size
  canvas = document.getElementById('inference');
  ctx = canvas.getContext('2d');

  canvas.width = window.innerWidth;
  canvas.height = Math.round(window.innerHeight / 3);

  w = Math.round(ctx.canvas.width);
  h = Math.round(ctx.canvas.height);

  // initialize
  cppn = tf.tidy(() => { return build(h, w) });
  input = tf.variable(cppn[0]);
  model = cppn[1];
  //dis = tf.variable(tf.ones([1, h, w, 8]).mul(tf.randomUniform([8], -2, 2)).sub(zstart));
  //dis.assign(dis.div(tf.scalar(fps * 15.)));  // take 15s from z1 to z2
  //tf.tidy(iter);
  zstart = tf.variable(tf.ones([1, h, w, 8]).mul(tf.randomUniform([8], -2, 2)));
  img = tf.variable(model.predict(tf.concat([input, zstart], -1)).squeeze());
  img.assign(tf.sigmoid(img.sub(tf.mean(img)).add(1)));
  //step = 0;
  //ctx.fillRect(0, 0, canvas.width, canvas.height);
  tf.browser.toPixels(img, canvas);
};

function iter() {
  zstart.assign(tf.ones([1, h, w, 8]).mul(tf.randomUniform([8], -2, 2)));
  img.assign(tf.variable(model.predict(tf.concat([input, zstart], -1)).squeeze()));
  img.assign(tf.sigmoid(img.sub(tf.mean(img)).add(1)));
  //step = 0;
  //ctx.fillRect(0, 0, canvas.width, canvas.height);
}


function draw() {
  tf.tidy(iter);
  tf.browser.toPixels(img, canvas);
}