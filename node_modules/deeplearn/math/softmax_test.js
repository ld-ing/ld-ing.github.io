"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('regular test', function (math) {
            var y = math.softmax(ndarray_1.Array1D.new([2, 1, 3]));
            test_util.expectArraysClose(y, [0.24472847, 0.09003057, 0.66524095]);
            test_util.expectNumbersClose(y.get(0) + y.get(1) + y.get(2), 1);
        });
        it('overflow', function (math) {
            var y = math.softmax(ndarray_1.Array1D.new([1000, 1000]));
            test_util.expectArraysClose(y, [0.5, 0.5]);
        });
        it('underflow', function (math) {
            var y = math.softmax(ndarray_1.Array1D.new([-1000, -1000]));
            test_util.expectArraysClose(y, [0.5, 0.5]);
        });
        it('Huge difference between probabilities', function (math) {
            var y = math.softmax(ndarray_1.Array1D.new([-1000, +1000]));
            test_util.expectArraysClose(y, [0, 1]);
        });
        it('Propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([2, 1, NaN]);
            var y = math.softmax(a);
            test_util.expectArraysClose(y, [NaN, NaN, NaN]);
        });
        it('2D, dim=1', function (math) {
            var y = math.softmax(ndarray_1.Array2D.new([2, 3], [[2, 1, 3], [1, 3, 2]]), 1);
            var expected = [
                0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
            ];
            expect(y.rank).toBe(2);
            test_util.expectArraysClose(y, expected);
        });
        it('2D, implicit dim=1', function (math) {
            var y = math.softmax(ndarray_1.Array2D.new([2, 3], [[2, 1, 3], [1, 3, 2]]));
            var expected = [
                0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
            ];
            expect(y.rank).toBe(2);
            test_util.expectArraysClose(y, expected);
        });
        it('2D, dim=0 throws error', function (math) {
            var f = function () {
                math.softmax(ndarray_1.Array2D.new([2, 3], [[2, 1, 3], [1, 3, 2]]), 0);
            };
            expect(f).toThrowError();
        });
        it('1D gradient', function (math) {
            var x = ndarray_1.Array1D.new([10, 0, -1]);
            var y = math.softmax(x);
            var dy = ndarray_1.Array1D.new([1, 2, 3]);
            var vjp = math.vjp(function () { return math.softmax(x); }, { x: x }, dy);
            var totalSum = math.sum(math.multiply(dy, y));
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, [
                (dy.get(0) - totalSum.get()) * y.get(0),
                (dy.get(1) - totalSum.get()) * y.get(1),
                (dy.get(2) - totalSum.get()) * y.get(2)
            ]);
        });
        it('2D gradient', function (math) {
            var x = ndarray_1.Array2D.new([2, 3], [10, 0, -1, 5, 4, 3]);
            var y = math.softmax(x);
            var dy = ndarray_1.Array2D.new([2, 3], [3, 2, 1, 1, 2, 3]);
            var vjp = math.vjp(function () { return math.softmax(x); }, { x: x }, dy);
            var axis = -1;
            var totalSum = math.sum(math.multiplyStrict(dy, y), axis);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, [
                (dy.get(0, 0) - totalSum.get(0)) * y.get(0, 0),
                (dy.get(0, 1) - totalSum.get(0)) * y.get(0, 1),
                (dy.get(0, 2) - totalSum.get(0)) * y.get(0, 2),
                (dy.get(1, 0) - totalSum.get(1)) * y.get(1, 0),
                (dy.get(1, 1) - totalSum.get(1)) * y.get(1, 1),
                (dy.get(1, 2) - totalSum.get(1)) * y.get(1, 2)
            ]);
        });
    };
    test_util.describeMathCPU('softmax', [tests]);
    test_util.describeMathGPU('softmax', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('1D', function (math) {
            var logits = ndarray_1.Array1D.new([1, 2, 3]);
            var label = ndarray_1.Array1D.new([0.3, 0.6, 0.1]);
            var softmaxLogits = math.softmax(logits);
            var y = math.softmaxCrossEntropy(label, logits);
            expect(y.shape).toEqual([]);
            test_util.expectNumbersClose(y.get(), -Math.log(softmaxLogits.get(0)) * label.get(0) +
                -Math.log(softmaxLogits.get(1)) * label.get(1) +
                -Math.log(softmaxLogits.get(2)) * label.get(2));
        });
        it('2D implicit dim', function (math) {
            var logits = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var label = ndarray_1.Array2D.new([2, 3], [0.3, 0.6, 0.1, 0.2, 0.3, 0.5]);
            var softmaxLogits = math.softmax(logits);
            var y = math.softmaxCrossEntropy(label, logits);
            expect(y.shape).toEqual([2]);
            test_util.expectArraysClose(y, [
                -Math.log(softmaxLogits.get(0, 0)) * label.get(0, 0) +
                    -Math.log(softmaxLogits.get(0, 1)) * label.get(0, 1) +
                    -Math.log(softmaxLogits.get(0, 2)) * label.get(0, 2),
                -Math.log(softmaxLogits.get(1, 0)) * label.get(1, 0) +
                    -Math.log(softmaxLogits.get(1, 1)) * label.get(1, 1) +
                    -Math.log(softmaxLogits.get(1, 2)) * label.get(1, 2)
            ]);
        });
        it('2D, dim=1', function (math) {
            var logits = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var label = ndarray_1.Array2D.new([2, 3], [0.3, 0.6, 0.1, 0.2, 0.3, 0.5]);
            var dim = 1;
            var softmaxLogits = math.softmax(logits, dim);
            var y = math.softmaxCrossEntropy(label, logits, dim);
            expect(y.shape).toEqual([2]);
            test_util.expectArraysClose(y, [
                -Math.log(softmaxLogits.get(0, 0)) * label.get(0, 0) +
                    -Math.log(softmaxLogits.get(0, 1)) * label.get(0, 1) +
                    -Math.log(softmaxLogits.get(0, 2)) * label.get(0, 2),
                -Math.log(softmaxLogits.get(1, 0)) * label.get(1, 0) +
                    -Math.log(softmaxLogits.get(1, 1)) * label.get(1, 1) +
                    -Math.log(softmaxLogits.get(1, 2)) * label.get(1, 2)
            ]);
        });
        it('2D, dim=0 throws error', function (math) {
            var logits = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var label = ndarray_1.Array2D.new([2, 3], [0.3, 0.6, 0.1, 0.2, 0.3, 0.5]);
            var dim = 0;
            expect(function () { return math.softmaxCrossEntropy(label, logits, dim); }).toThrowError();
        });
        it('Propagates NaNs', function (math) {
            var logits = ndarray_1.Array1D.new([1, 2, NaN]);
            var label = ndarray_1.Array1D.new([0.3, 0.6, 0.1]);
            var y = math.softmaxCrossEntropy(label, logits);
            expect(y.shape).toEqual([]);
            test_util.expectArraysClose(y, [NaN]);
        });
        it('1D gradient', function (math) {
            var logits = ndarray_1.Array1D.new([1, 2, 3]);
            var labels = ndarray_1.Array1D.new([0.3, 0.6, 0.1]);
            var softmaxLogits = math.softmax(logits);
            var dy = ndarray_1.Scalar.new(2);
            var vjp = math.vjp(function () { return math.softmaxCrossEntropy(labels, logits); }, { labels: labels, logits: logits }, dy);
            expect(vjp.logits.shape).toEqual(logits.shape);
            expect(vjp.labels.shape).toEqual(labels.shape);
            test_util.expectArraysClose(vjp.logits, [
                dy.get() * (softmaxLogits.get(0) - labels.get(0)),
                dy.get() * (softmaxLogits.get(1) - labels.get(1)),
                dy.get() * (softmaxLogits.get(2) - labels.get(2))
            ]);
            test_util.expectArraysClose(vjp.labels, [
                dy.get() * (labels.get(0) - softmaxLogits.get(0)),
                dy.get() * (labels.get(1) - softmaxLogits.get(1)),
                dy.get() * (labels.get(2) - softmaxLogits.get(2))
            ]);
        });
        it('2D gradient', function (math) {
            var logits = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var labels = ndarray_1.Array2D.new([2, 3], [0.3, 0.6, 0.1, .2, .3, .5]);
            var softmaxLogits = math.softmax(logits);
            var dy = ndarray_1.Array1D.new([2, 4]);
            var vjp = math.vjp(function () { return math.softmaxCrossEntropy(labels, logits); }, { labels: labels, logits: logits }, dy);
            expect(vjp.logits.shape).toEqual(logits.shape);
            expect(vjp.labels.shape).toEqual(labels.shape);
            test_util.expectArraysClose(vjp.logits, [
                dy.get(0) * (softmaxLogits.get(0, 0) - labels.get(0, 0)),
                dy.get(0) * (softmaxLogits.get(0, 1) - labels.get(0, 1)),
                dy.get(0) * (softmaxLogits.get(0, 2) - labels.get(0, 2)),
                dy.get(1) * (softmaxLogits.get(1, 0) - labels.get(1, 0)),
                dy.get(1) * (softmaxLogits.get(1, 1) - labels.get(1, 1)),
                dy.get(1) * (softmaxLogits.get(1, 2) - labels.get(1, 2))
            ]);
            test_util.expectArraysClose(vjp.labels, [
                dy.get(0) * (labels.get(0, 0) - softmaxLogits.get(0, 0)),
                dy.get(0) * (labels.get(0, 1) - softmaxLogits.get(0, 1)),
                dy.get(0) * (labels.get(0, 2) - softmaxLogits.get(0, 2)),
                dy.get(1) * (labels.get(1, 0) - softmaxLogits.get(1, 0)),
                dy.get(1) * (labels.get(1, 1) - softmaxLogits.get(1, 1)),
                dy.get(1) * (labels.get(1, 2) - softmaxLogits.get(1, 2))
            ]);
        });
    };
    test_util.describeMathCPU('softmaxCrossEntropy', [tests]);
    test_util.describeMathGPU('softmaxCrossEntropy', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=softmax_test.js.map