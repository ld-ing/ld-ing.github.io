// Li Ding, July 2018
// Source:
// http://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf
// http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/
// https://js.tensorflow.org/


function draw() {
  var canvas = document.getElementById('inference');
  var ctx = canvas.getContext('2d');
  var slider = document.getElementById('cppnslider');

  // initialize
  ctx.canvas.width = window.innerWidth;
  ctx.canvas.height = Math.round(window.innerHeight/2);
  var resolution = slider.value;

  var ratio = ctx.canvas.height / ctx.canvas.width;
  var w = Math.round(ctx.canvas.width * resolution / 100);
  var h = Math.round(w*ratio);

  var cppn = build(h,w);
  var input = cppn[0];
  var model = cppn[1];

  var zstart = tf.ones([1,h,w,8]).mul(tf.randomUniform([8],-1,1));
  var zend = tf.ones([1,h,w,8]).mul(tf.randomUniform([8],-1,1));
  var dis = zend.sub(zstart);
  dis = dis.div(tf.scalar(100.));  // take 100 steps from z1 to z2
  var step = 0;
  var img;

  // iterator
  window.setInterval(function(){
    if (resolution==slider.value && 
    ctx.canvas.width == window.innerWidth && 
    ctx.canvas.height == Math.round(window.innerHeight/2)) {
      img = model.predict(tf.concat([input,zstart],-1));
      img = tf.image.resizeBilinear(img,
          [ctx.canvas.height, ctx.canvas.width]);
      tf.toPixels(img.squeeze(), canvas);
      zstart = zstart.add(dis);
      step ++;
      if (step > 100){
        zend = tf.ones([1,h,w,8]).mul(tf.randomUniform([8],-1,1));
        dis = zend.sub(zstart);
        dis = dis.div(tf.scalar(100.));
        step = 0;
      };
    } else {
      ctx.canvas.width = window.innerWidth;
      ctx.canvas.height = Math.round(window.innerHeight/2);
      resolution = slider.value;
    
      ratio = ctx.canvas.height / ctx.canvas.width;
      w = Math.round(ctx.canvas.width * resolution / 100);
      h = Math.round(w*ratio);
    
      cppn = build(h,w);
      input = cppn[0];
      model = cppn[1];
    
      zstart = tf.ones([1,h,w,8]).mul(tf.randomUniform([8],-1,1));
      zend = tf.ones([1,h,w,8]).mul(tf.randomUniform([8],-1,1));
      dis = zend.sub(zstart);
      dis = dis.div(tf.scalar(100.));  // take 100 steps from z1 to z2
      step = 0;
    };
  }, 100);  // .1s per update
};

function build(h,w) {
  var x = tf.range(-1, 1, 2/w);
  x = x.tile([h]).reshape([h,w,1]);
  var y = tf.range(-h/w, h/w, 2/w);
  y = y.tile([w]).reshape([w,h,1]).transpose([1,0,2]);
  var r = tf.sqrt(x.pow(2).add(y.pow(2)));
  const input = tf.concat([x,y,r],-1).reshape([1,h,w,3]);
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [h, w, 11],
    kernelSize: 3,
    filters: 32,
    strides: 1,
    activation: 'tanh',
    kernelInitializer: tf.initializers.randomNormal({mean:0, stddev:1}),
    useBias: true
  }));
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 32,
    strides: 1,
    activation: 'tanh',
    kernelInitializer: tf.initializers.randomNormal({mean:0, stddev:1}),
    useBias: true
  }));
  model.add(tf.layers.conv2d({
    kernelSize: 1,
    filters: 32,
    strides: 1,
    activation: 'tanh',
    kernelInitializer: tf.initializers.randomNormal({mean:0, stddev:1}),
    useBias: true
  }));
  model.add(tf.layers.conv2d({
    kernelSize: 1,
    filters: 1,
    strides: 1,
    activation: 'sigmoid',
    kernelInitializer: tf.initializers.randomNormal({mean:0, stddev:1}),
    useBias: true
  }));
  return [input, model];
};
