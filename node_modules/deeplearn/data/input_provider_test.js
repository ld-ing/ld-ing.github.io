"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var math_1 = require("../math/math");
var ndarray_1 = require("../math/ndarray");
var input_provider_1 = require("./input_provider");
describe('InCPUMemoryShuffledInputProviderBuilder', function () {
    var math;
    beforeEach(function () {
        var safeMode = false;
        math = new math_1.NDArrayMath('cpu', safeMode);
        environment_1.ENV.setMath(math);
    });
    afterEach(function () {
        environment_1.ENV.reset();
    });
    it('ensure inputs stay in sync', function () {
        var x1s = [ndarray_1.Scalar.new(1), ndarray_1.Scalar.new(2), ndarray_1.Scalar.new(3)];
        var x2s = [ndarray_1.Scalar.new(10), ndarray_1.Scalar.new(20), ndarray_1.Scalar.new(30)];
        var shuffledInputProvider = new input_provider_1.InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]);
        var _a = shuffledInputProvider.getInputProviders(), x1provider = _a[0], x2provider = _a[1];
        var seenNumbers = {};
        for (var i = 0; i < x1s.length; i++) {
            var x1 = x1provider.getNextCopy(math);
            var x2 = x2provider.getNextCopy(math);
            expect(x1.get() * 10).toEqual(x2.get());
            seenNumbers[x1.get()] = true;
            seenNumbers[x2.get()] = true;
        }
        var expectedSeenNumbers = [1, 2, 3, 10, 20, 30];
        for (var i = 0; i < expectedSeenNumbers.length; i++) {
            expect(seenNumbers[expectedSeenNumbers[i]]).toEqual(true);
        }
    });
    it('different number of examples', function () {
        var x1s = [ndarray_1.Scalar.new(1), ndarray_1.Scalar.new(2)];
        var x2s = [ndarray_1.Scalar.new(10), ndarray_1.Scalar.new(20), ndarray_1.Scalar.new(30)];
        expect(function () { return new input_provider_1.InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]); })
            .toThrowError();
    });
    it('different shapes within input', function () {
        var x1s = [ndarray_1.Scalar.new(1), ndarray_1.Array1D.new([1, 2])];
        var x2s = [ndarray_1.Scalar.new(10), ndarray_1.Scalar.new(20), ndarray_1.Scalar.new(30)];
        expect(function () { return new input_provider_1.InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]); })
            .toThrowError();
    });
});
describe('InGPUMemoryShuffledInputProviderBuilder', function () {
    var math;
    beforeEach(function () {
        var safeMode = false;
        math = new math_1.NDArrayMath('webgl', safeMode);
        environment_1.ENV.setMath(math);
    });
    afterEach(function () {
        environment_1.ENV.reset();
    });
    it('ensure inputs stay in sync', function () {
        var x1s = [ndarray_1.Scalar.new(1), ndarray_1.Scalar.new(2), ndarray_1.Scalar.new(3)];
        var x2s = [ndarray_1.Scalar.new(10), ndarray_1.Scalar.new(20), ndarray_1.Scalar.new(30)];
        var shuffledInputProvider = new input_provider_1.InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]);
        var _a = shuffledInputProvider.getInputProviders(), x1provider = _a[0], x2provider = _a[1];
        var seenNumbers = {};
        for (var i = 0; i < x1s.length; i++) {
            var x1 = x1provider.getNextCopy(math);
            var x2 = x2provider.getNextCopy(math);
            expect(x1.get() * 10).toEqual(x2.get());
            seenNumbers[x1.get()] = true;
            seenNumbers[x2.get()] = true;
            x1provider.disposeCopy(math, x1);
            x2provider.disposeCopy(math, x1);
        }
        var expectedSeenNumbers = [1, 2, 3, 10, 20, 30];
        for (var i = 0; i < expectedSeenNumbers.length; i++) {
            expect(seenNumbers[expectedSeenNumbers[i]]).toEqual(true);
        }
    });
    it('different number of examples', function () {
        var x1s = [ndarray_1.Scalar.new(1), ndarray_1.Scalar.new(2)];
        var x2s = [ndarray_1.Scalar.new(10), ndarray_1.Scalar.new(20), ndarray_1.Scalar.new(30)];
        expect(function () { return new input_provider_1.InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]); })
            .toThrowError();
        x1s.forEach(function (x1) {
            x1.dispose();
        });
        x2s.forEach(function (x2) {
            x2.dispose();
        });
    });
    it('different shapes within input', function () {
        var x1s = [ndarray_1.Scalar.new(1), ndarray_1.Array1D.new([1, 2])];
        var x2s = [ndarray_1.Scalar.new(10), ndarray_1.Scalar.new(20), ndarray_1.Scalar.new(30)];
        expect(function () { return new input_provider_1.InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]); })
            .toThrowError();
        x1s.forEach(function (x1) {
            x1.dispose();
        });
        x2s.forEach(function (x2) {
            x2.dispose();
        });
    });
});
//# sourceMappingURL=input_provider_test.js.map