// CPPN step generator and render on canvas
// Author: Li Ding
// Last Update: Oct. 2018
// Source:
// http://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf
// http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/
// https://js.tensorflow.org/


// prevent memory overflow
'use strict';

// on load function
function draw() {
  let canvas = document.getElementById('inference');
  let ctx = canvas.getContext('2d');
  let slider = document.getElementById('cppnslider');

  let fps = 0;
  let w, h, cppn, model, step, input, zstart, dis, img;

  // make canvas to fit window size
  ctx.canvas.width = window.innerWidth;
  ctx.canvas.height = Math.round(window.innerHeight / 3);
  fps = slider.value;

  w = Math.round(ctx.canvas.width);
  h = Math.round(ctx.canvas.height);

  // build cppn model
  function build(h, w) {
    let x = tf.range(0, 4 * w, 4);
    x = x.tile([h]).reshape([h, w, 1]);
    x = x.div(w).sub(2)
    let y = tf.range(0, 2 * h, 2);
    y = y.tile([w]).reshape([w, h, 1]).transpose([1, 0, 2]);
    y = y.div(h).sub(1)
    const r = tf.sqrt(x.pow(2).add(y.pow(2)));
    const input = tf.concat([x, y, r], -1).reshape([1, h, w, 3]);
    let model = tf.sequential();
    model.add(tf.layers.conv2d({
      inputShape: [h, w, 11],
      kernelSize: 1,
      filters: 16,
      strides: 1,
      activation: 'tanh',
      kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 2 }),
      useBias: true
    }));
    model.add(tf.layers.conv2d({
      kernelSize: 1,
      filters: 16,
      strides: 1,
      activation: 'tanh',
      kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 2 }),
      useBias: true
    }));
    model.add(tf.layers.conv2d({
      kernelSize: 1,
      filters: 16,
      strides: 1,
      activation: 'tanh',
      kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 2 }),
      useBias: true
    }));
    model.add(tf.layers.conv2d({
      kernelSize: 1,
      filters: 1,
      strides: 1,
      activation: 'sigmoid',
      kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 1 }),
      useBias: false
    }));
    return [input, model];
  };

  // change the canvas by step
  function iter() {
    img.assign(model.predict(tf.concat([input, zstart], -1)).squeeze());
    zstart.assign(zstart.add(dis));
    step++;
    if (step > fps * 10.) {
      dis.assign(tf.ones([1, h, w, 8]).mul(tf.randomUniform([8], -1, 1)).sub(zstart));
      dis.assign(dis.div(tf.scalar(fps * 10.)));  // take 10s from z1 to z2
      step = 0;
    };
  }

  // reset image change rate if fps slider is changed
  function change() {
    dis.assign(tf.ones([1, h, w, 8]).mul(tf.randomUniform([8], -1, 1)).sub(zstart));
    dis.assign(dis.div(tf.scalar(fps * 10.)));
    step = 0;
    img.assign(model.predict(tf.concat([input, zstart], -1)).squeeze());
  }

  // change the canvas by step
  function move() {
    //console.log(tf.memory())
    document.getElementById("fps").innerHTML = "FPS: " + slider.value;
    if (fps != slider.value) {
      fps = slider.value;
      window.clearInterval(interval)
      interval = window.setInterval(move, 1000. / fps);
      tf.tidy(change)
    }
    if (ctx.canvas.width == window.innerWidth &&
      ctx.canvas.height == Math.round(window.innerHeight / 3)) {
      tf.tidy(iter);
      tf.toPixels(img, canvas);
    } else {
      tf.dispose()  // not working actually
      ctx.canvas.width = window.innerWidth;
      ctx.canvas.height = Math.round(window.innerHeight / 3);

      w = Math.round(ctx.canvas.width);
      h = Math.round(ctx.canvas.height);

      cppn = tf.tidy(() => { return build(h, w) });
      input = tf.variable(cppn[0]);
      model = cppn[1]

      zstart = tf.variable(tf.ones([1, h, w, 8]).mul(tf.randomUniform([8], -1, 1)));
      dis = tf.variable(tf.ones([1, h, w, 8]).mul(tf.randomUniform([8], -1, 1)).sub(zstart));
      dis.assign(dis.div(tf.scalar(fps * 10.)));  // take 10s from z1 to z2
      img = tf.variable(model.predict(tf.concat([input, zstart], -1)).squeeze());
      step = 0;
    };
  }

  // initialize
  cppn = tf.tidy(() => { return build(h, w) });
  input = tf.variable(cppn[0]);
  model = cppn[1];
  zstart = tf.variable(tf.ones([1, h, w, 8]).mul(tf.randomUniform([8], -1, 1)));
  dis = tf.variable(tf.ones([1, h, w, 8]).mul(tf.randomUniform([8], -1, 1)).sub(zstart));
  dis.assign(dis.div(tf.scalar(fps * 10.)));  // take 10s from z1 to z2
  img = tf.variable(model.predict(tf.concat([input, zstart], -1)).squeeze());
  step = 0;

  // iterate forever
  var interval = window.setInterval(move, 1000. / fps);
};

