"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("./environment");
var backend_cpu_1 = require("./math/backends/backend_cpu");
var backend_webgl_1 = require("./math/backends/backend_webgl");
var math_1 = require("./math/math");
var ndarray_1 = require("./math/ndarray");
var util = require("./util");
exports.TEST_EPSILON = 1e-2;
function mean(values) {
    var sum = 0;
    for (var i = 0; i < values.length; i++) {
        sum += values[i];
    }
    return sum / values.length;
}
exports.mean = mean;
function standardDeviation(values, mean) {
    var squareDiffSum = 0;
    for (var i = 0; i < values.length; i++) {
        var diff = values[i] - mean;
        squareDiffSum += diff * diff;
    }
    return Math.sqrt(squareDiffSum / values.length);
}
exports.standardDeviation = standardDeviation;
function kurtosis(values) {
    var valuesMean = mean(values);
    var n = values.length;
    var sum2 = 0;
    var sum4 = 0;
    for (var i = 0; i < n; i++) {
        var v = values[i] - valuesMean;
        sum2 += Math.pow(v, 2);
        sum4 += Math.pow(v, 4);
    }
    return (1 / n) * sum4 / Math.pow((1 / n) * sum2, 2);
}
exports.kurtosis = kurtosis;
function skewness(values) {
    var valuesMean = mean(values);
    var n = values.length;
    var sum2 = 0;
    var sum3 = 0;
    for (var i = 0; i < n; i++) {
        var v = values[i] - valuesMean;
        sum2 += Math.pow(v, 2);
        sum3 += Math.pow(v, 3);
    }
    return (1 / n) * sum3 / Math.pow((1 / (n - 1)) * sum2, 3 / 2);
}
exports.skewness = skewness;
function jarqueBeraNormalityTest(a) {
    var values;
    if (a instanceof ndarray_1.NDArray) {
        values = a.dataSync();
    }
    else {
        values = a;
    }
    var n = values.length;
    var s = skewness(values);
    var k = kurtosis(values);
    var jb = n / 6 * (Math.pow(s, 2) + 0.25 * Math.pow(k - 3, 2));
    var CHI_SQUARE_2DEG = 5.991;
    if (jb > CHI_SQUARE_2DEG) {
        throw new Error("Invalid p-value for JB: " + jb);
    }
}
exports.jarqueBeraNormalityTest = jarqueBeraNormalityTest;
function expectArrayInMeanStdRange(actual, expectedMean, expectedStdDev, epsilon) {
    if (epsilon === void 0) { epsilon = exports.TEST_EPSILON; }
    var actualValues;
    if (actual instanceof ndarray_1.NDArray) {
        actualValues = actual.dataSync();
    }
    else {
        actualValues = actual;
    }
    var actualMean = mean(actualValues);
    expectNumbersClose(actualMean, expectedMean, epsilon);
    expectNumbersClose(standardDeviation(actualValues, actualMean), expectedStdDev, epsilon);
}
exports.expectArrayInMeanStdRange = expectArrayInMeanStdRange;
function expectArraysClose(actual, expected, epsilon) {
    if (epsilon === void 0) { epsilon = exports.TEST_EPSILON; }
    if (!(actual instanceof ndarray_1.NDArray) && !(expected instanceof ndarray_1.NDArray)) {
        var aType = actual.constructor.name;
        var bType = expected.constructor.name;
        if (aType !== bType) {
            throw new Error("Arrays are of different type actual: " + aType + " " +
                ("vs expected: " + bType));
        }
    }
    else if (actual instanceof ndarray_1.NDArray && expected instanceof ndarray_1.NDArray) {
        if (actual.dtype !== expected.dtype) {
            throw new Error("Arrays are of different type actual: " + actual.dtype + " " +
                ("vs expected: " + expected.dtype + "."));
        }
        if (!util.arraysEqual(actual.shape, expected.shape)) {
            throw new Error("Arrays are of different shape actual: " + actual.shape + " " +
                ("vs expected: " + expected.shape + "."));
        }
    }
    var actualValues;
    var expectedValues;
    if (actual instanceof ndarray_1.NDArray) {
        actualValues = actual.dataSync();
    }
    else {
        actualValues = actual;
    }
    if (expected instanceof ndarray_1.NDArray) {
        expectedValues = expected.dataSync();
    }
    else {
        expectedValues = expected;
    }
    if (actualValues.length !== expectedValues.length) {
        throw new Error("Arrays have different lengths actual: " + actualValues.length + " vs " +
            ("expected: " + expectedValues.length + ".\n") +
            ("Actual:   " + actualValues + ".\n") +
            ("Expected: " + expectedValues + "."));
    }
    for (var i = 0; i < expectedValues.length; ++i) {
        var a = actualValues[i];
        var e = expectedValues[i];
        if (!areClose(a, Number(e), epsilon)) {
            throw new Error("Arrays differ: actual[" + i + "] = " + a + ", expected[" + i + "] = " + e + ".\n" +
                ("Actual:   " + actualValues + ".\n") +
                ("Expected: " + expectedValues + "."));
        }
    }
}
exports.expectArraysClose = expectArraysClose;
function expectArraysEqual(actual, expected) {
    return expectArraysClose(actual, expected, 0);
}
exports.expectArraysEqual = expectArraysEqual;
function expectNumbersClose(a, e, epsilon) {
    if (epsilon === void 0) { epsilon = exports.TEST_EPSILON; }
    if (!areClose(a, e, epsilon)) {
        throw new Error("Numbers differ: actual === " + a + ", expected === " + e);
    }
}
exports.expectNumbersClose = expectNumbersClose;
function areClose(a, e, epsilon) {
    if (isNaN(a) && isNaN(e)) {
        return true;
    }
    if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
        return false;
    }
    return true;
}
function expectValuesInRange(actual, low, high) {
    var actualVals;
    if (actual instanceof ndarray_1.NDArray) {
        actualVals = actual.dataSync();
    }
    else {
        actualVals = actual;
    }
    for (var i = 0; i < actualVals.length; i++) {
        if (actualVals[i] < low || actualVals[i] > high) {
            throw new Error("Value out of range:" + actualVals[i] + " low: " + low + ", high: " + high);
        }
    }
}
exports.expectValuesInRange = expectValuesInRange;
function randomArrayInRange(n, minValue, maxValue) {
    var v = new Float32Array(n);
    var range = maxValue - minValue;
    for (var i = 0; i < n; ++i) {
        v[i] = (Math.random() * range) + minValue;
    }
    return v;
}
exports.randomArrayInRange = randomArrayInRange;
function makeIdentity(n) {
    var i = new Float32Array(n * n);
    for (var j = 0; j < n; ++j) {
        i[(j * n) + j] = 1;
    }
    return i;
}
exports.makeIdentity = makeIdentity;
function cpuMultiplyMatrix(a, aRow, aCol, b, bRow, bCol) {
    var result = new Float32Array(aRow * bCol);
    for (var r = 0; r < aRow; ++r) {
        var aOffset = (r * aCol);
        var cOffset = (r * bCol);
        for (var c = 0; c < bCol; ++c) {
            var d = 0;
            for (var k = 0; k < aCol; ++k) {
                d += a[aOffset + k] * b[(k * bCol) + c];
            }
            result[cOffset + c] = d;
        }
    }
    return result;
}
exports.cpuMultiplyMatrix = cpuMultiplyMatrix;
function cpuDotProduct(a, b) {
    if (a.length !== b.length) {
        throw new Error('cpuDotProduct: incompatible vectors.');
    }
    var d = 0;
    for (var i = 0; i < a.length; ++i) {
        d += a[i] * b[i];
    }
    return d;
}
exports.cpuDotProduct = cpuDotProduct;
function describeMathCPU(name, tests, featuresList) {
    var testNameBase = 'CPU: math.' + name;
    describeWithFeaturesAndExecutor(testNameBase, tests, function (testName, tests, features) { return executeMathTests(testName, tests, function () {
        var safeMode = true;
        return new math_1.NDArrayMath(new backend_cpu_1.MathBackendCPU(), safeMode);
    }, features); }, featuresList);
}
exports.describeMathCPU = describeMathCPU;
function describeMathGPU(name, tests, featuresList) {
    var testNameBase = 'WebGL: math.' + name;
    describeWithFeaturesAndExecutor(testNameBase, tests, function (testName, tests, features) { return executeMathTests(testName, tests, function () {
        var safeMode = true;
        return new math_1.NDArrayMath(new backend_webgl_1.MathBackendWebGL(), safeMode);
    }, features); }, featuresList);
}
exports.describeMathGPU = describeMathGPU;
function describeCustom(name, tests, featuresList, customBeforeEach, customAfterEach) {
    describeWithFeaturesAndExecutor(name, [tests], function (testName, tests, features) { return executeTests(testName, tests, features, customBeforeEach, customAfterEach); }, featuresList);
}
exports.describeCustom = describeCustom;
function describeWithFeaturesAndExecutor(testNameBase, tests, executor, featuresList) {
    if (featuresList != null) {
        featuresList.forEach(function (features) {
            var testName = testNameBase + ' ' + JSON.stringify(features);
            executor(testName, tests, features);
        });
    }
    else {
        executor(testNameBase, tests);
    }
}
function resolveTestFuncPromise(testFunc) {
    return function (done) {
        var result = testFunc();
        if (result instanceof Promise) {
            result.then(done, function (e) {
                fail(e);
                done();
            });
        }
        else {
            done();
        }
    };
}
var PROMISE_IT = function (name, testFunc) {
    it(name, resolveTestFuncPromise(testFunc));
};
var PROMISE_FIT = function (name, testFunc) {
    fit(name, resolveTestFuncPromise(testFunc));
};
var PROMISE_XIT = function (name, testFunc) {
    xit(name, resolveTestFuncPromise(testFunc));
};
function executeMathTests(testName, tests, mathFactory, features) {
    var math;
    var customBeforeEach = function () {
        math = mathFactory();
        environment_1.ENV.setMath(math);
        math.startScope();
    };
    var customAfterEach = function () {
        math.endScope(null);
        math.dispose();
    };
    var customIt = function (name, testFunc) {
        PROMISE_IT(name, function () { return testFunc(math); });
    };
    var customFit = function (name, testFunc) {
        PROMISE_FIT(name, function () { return testFunc(math); });
    };
    var customXit = function (name, testFunc) {
        PROMISE_XIT(name, function () { return testFunc(math); });
    };
    executeTests(testName, tests, features, customBeforeEach, customAfterEach, customIt, customFit, customXit);
}
exports.executeMathTests = executeMathTests;
function executeTests(testName, tests, features, customBeforeEach, customAfterEach, customIt, customFit, customXit) {
    if (customIt === void 0) { customIt = PROMISE_IT; }
    if (customFit === void 0) { customFit = PROMISE_FIT; }
    if (customXit === void 0) { customXit = PROMISE_XIT; }
    describe(testName, function () {
        beforeEach(function () {
            if (features != null) {
                environment_1.ENV.setFeatures(features);
                environment_1.ENV.addCustomBackend('webgl', function () { return new backend_webgl_1.MathBackendWebGL(); });
                environment_1.ENV.addCustomBackend('cpu', function () { return new backend_cpu_1.MathBackendCPU(); });
            }
            if (customBeforeEach != null) {
                customBeforeEach();
            }
        });
        afterEach(function () {
            if (customAfterEach != null) {
                customAfterEach();
            }
            if (features != null) {
                environment_1.ENV.reset();
            }
        });
        tests.forEach(function (test) { return test(customIt, customFit, customXit); });
    });
}
function assertIsNan(val, dtype) {
    if (!util.isValNaN(val, dtype)) {
        throw new Error("Value " + val + " does not represent NaN for dtype " + dtype);
    }
}
exports.assertIsNan = assertIsNan;
//# sourceMappingURL=test_util.js.map