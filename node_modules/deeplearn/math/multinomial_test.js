"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
var tests = function (it) {
    var NUM_SAMPLES = 10000;
    var EPSILON = 0.05;
    it('Flip a fair coin and check bounds', function (math) {
        var probs = ndarray_1.Array1D.new([0.5, 0.5]);
        var result = math.multinomial(probs, NUM_SAMPLES);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync(), 2);
        test_util.expectArraysClose(outcomeProbs, [0.5, 0.5], EPSILON);
    });
    it('Flip a two-sided coin with 100% of heads', function (math) {
        var probs = ndarray_1.Array1D.new([1, 0]);
        var result = math.multinomial(probs, NUM_SAMPLES);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync(), 2);
        test_util.expectArraysClose(outcomeProbs, [1, 0], EPSILON);
    });
    it('Flip a two-sided coin with 100% of tails', function (math) {
        var probs = ndarray_1.Array1D.new([0, 1]);
        var result = math.multinomial(probs, NUM_SAMPLES);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync(), 2);
        test_util.expectArraysClose(outcomeProbs, [0, 1], EPSILON);
    });
    it('Flip a single-sided coin throws error', function (math) {
        var probs = ndarray_1.Array1D.new([1]);
        expect(function () { return math.multinomial(probs, NUM_SAMPLES); }).toThrowError();
    });
    it('Flip a ten-sided coin and check bounds', function (math) {
        var numOutcomes = 10;
        var probs = dl.zeros([numOutcomes]);
        for (var i = 0; i < numOutcomes; ++i) {
            probs.set(1 / numOutcomes, i);
        }
        var result = math.multinomial(probs, NUM_SAMPLES);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync(), numOutcomes);
        expect(outcomeProbs.length).toBeLessThanOrEqual(numOutcomes);
    });
    it('Flip 3 three-sided coins, each coin is 100% biases', function (math) {
        var numOutcomes = 3;
        var probs = ndarray_1.Array2D.new([3, numOutcomes], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]);
        var result = math.multinomial(probs, NUM_SAMPLES);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([3, NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync().slice(0, NUM_SAMPLES), numOutcomes);
        test_util.expectArraysClose(outcomeProbs, [0, 0, 1], EPSILON);
        outcomeProbs = computeProbs(result.dataSync().slice(NUM_SAMPLES, 2 * NUM_SAMPLES), numOutcomes);
        test_util.expectArraysClose(outcomeProbs, [0, 1, 0], EPSILON);
        outcomeProbs =
            computeProbs(result.dataSync().slice(2 * NUM_SAMPLES), numOutcomes);
        test_util.expectArraysClose(outcomeProbs, [1, 0, 0], EPSILON);
    });
    it('passing Array3D throws error', function (math) {
        var probs = dl.zeros([3, 2, 2]);
        expect(function () { return math.multinomial(probs, 3); }).toThrowError();
    });
    function computeProbs(events, numOutcomes) {
        var counts = [];
        for (var i = 0; i < numOutcomes; ++i) {
            counts[i] = 0;
        }
        var numSamples = events.length;
        for (var i = 0; i < events.length; ++i) {
            counts[events[i]]++;
        }
        for (var i = 0; i < counts.length; i++) {
            counts[i] /= numSamples;
        }
        return counts;
    }
};
test_util.describeMathCPU('multinomial', [tests]);
test_util.describeMathGPU('multinomial', [tests], [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=multinomial_test.js.map