"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var util = require("../util");
var matmul_1 = require("./backends/types/matmul");
var ndarray_1 = require("./ndarray");
{
    var gpuTests = function (it) {
        it('scope returns NDArray', function (math) { return __awaiter(_this, void 0, void 0, function () {
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, math.scope(function () { return __awaiter(_this, void 0, void 0, function () {
                            var _this = this;
                            var a, b;
                            return __generator(this, function (_a) {
                                switch (_a.label) {
                                    case 0:
                                        a = ndarray_1.Array1D.new([1, 2, 3]);
                                        b = ndarray_1.Array1D.new([0, 0, 0]);
                                        expect(math.getNumArrays()).toBe(2);
                                        return [4, math.scope(function () { return __awaiter(_this, void 0, void 0, function () {
                                                var result;
                                                return __generator(this, function (_a) {
                                                    result = math.scope(function () {
                                                        b = math.addStrict(a, b);
                                                        b = math.addStrict(a, b);
                                                        b = math.addStrict(a, b);
                                                        return math.add(a, b);
                                                    });
                                                    expect(math.getNumArrays()).toBe(2 + 1);
                                                    test_util.expectArraysClose(result, [4, 8, 12]);
                                                    return [2];
                                                });
                                            }); })];
                                    case 1:
                                        _a.sent();
                                        expect(math.getNumArrays()).toBe(2);
                                        return [2];
                                }
                            });
                        }); })];
                    case 1:
                        _a.sent();
                        expect(math.getNumArrays()).toBe(0);
                        return [2];
                }
            });
        }); });
        it('multiple disposes does not affect num arrays', function (math) {
            expect(math.getNumArrays()).toBe(0);
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var b = ndarray_1.Array1D.new([1, 2, 3]);
            expect(math.getNumArrays()).toBe(2);
            a.dispose();
            a.dispose();
            expect(math.getNumArrays()).toBe(1);
            b.dispose();
            expect(math.getNumArrays()).toBe(0);
        });
        it('scope returns NDArray[]', function (math) { return __awaiter(_this, void 0, void 0, function () {
            var _this = this;
            var a, b;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        a = ndarray_1.Array1D.new([1, 2, 3]);
                        b = ndarray_1.Array1D.new([0, -1, 1]);
                        expect(math.getNumArrays()).toBe(2);
                        return [4, math.scope(function () { return __awaiter(_this, void 0, void 0, function () {
                                var result;
                                return __generator(this, function (_a) {
                                    result = math.scope(function () {
                                        math.add(a, b);
                                        return [math.add(a, b), math.subtract(a, b)];
                                    });
                                    expect(math.getNumArrays()).toBe(4);
                                    test_util.expectArraysClose(result[0], [1, 1, 4]);
                                    test_util.expectArraysClose(result[1], [1, 3, 2]);
                                    expect(math.getNumArrays()).toBe(4);
                                    return [2];
                                });
                            }); })];
                    case 1:
                        _a.sent();
                        expect(math.getNumArrays()).toBe(2);
                        a.dispose();
                        b.dispose();
                        expect(math.getNumArrays()).toBe(0);
                        return [2];
                }
            });
        }); });
        it('basic scope usage without return', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var b = ndarray_1.Array1D.new([0, 0, 0]);
            expect(math.getNumArrays()).toBe(2);
            math.scope(function () {
                b = math.addStrict(a, b);
                b = math.addStrict(a, b);
                b = math.addStrict(a, b);
                math.add(a, b);
            });
            expect(math.getNumArrays()).toBe(2);
        });
        it('scope returns Promise<NDArray>', function (math) { return __awaiter(_this, void 0, void 0, function () {
            var _this = this;
            var a, b;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        a = ndarray_1.Array1D.new([1, 2, 3]);
                        b = ndarray_1.Array1D.new([0, 0, 0]);
                        expect(math.getNumArrays()).toBe(2);
                        return [4, math.scope(function () { return __awaiter(_this, void 0, void 0, function () {
                                var result;
                                return __generator(this, function (_a) {
                                    result = math.scope(function () {
                                        var c = math.add(a, b);
                                        c = math.add(a, c);
                                        c = math.add(a, c);
                                        return math.add(a, c);
                                    });
                                    expect(math.getNumArrays()).toBe(3);
                                    test_util.expectArraysClose(result, [4, 8, 12]);
                                    return [2];
                                });
                            }); })];
                    case 1:
                        _a.sent();
                        expect(math.getNumArrays()).toBe(2);
                        a.dispose();
                        b.dispose();
                        expect(math.getNumArrays()).toBe(0);
                        return [2];
                }
            });
        }); });
        it('nested scope usage', function (math) { return __awaiter(_this, void 0, void 0, function () {
            var _this = this;
            var a, b;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        a = ndarray_1.Array1D.new([1, 2, 3]);
                        b = ndarray_1.Array1D.new([0, 0, 0]);
                        expect(math.getNumArrays()).toBe(2);
                        return [4, math.scope(function () { return __awaiter(_this, void 0, void 0, function () {
                                var result;
                                return __generator(this, function (_a) {
                                    result = math.scope(function () {
                                        b = math.addStrict(a, b);
                                        b = math.scope(function () {
                                            b = math.scope(function () {
                                                return math.addStrict(a, b);
                                            });
                                            expect(math.getNumArrays()).toBe(4);
                                            math.scope(function () {
                                                math.addStrict(a, b);
                                            });
                                            expect(math.getNumArrays()).toBe(4);
                                            return math.addStrict(a, b);
                                        });
                                        expect(math.getNumArrays()).toBe(4);
                                        return math.addStrict(a, b);
                                    });
                                    expect(math.getNumArrays()).toBe(3);
                                    test_util.expectArraysClose(result, [4, 8, 12]);
                                    return [2];
                                });
                            }); })];
                    case 1:
                        _a.sent();
                        expect(math.getNumArrays()).toBe(2);
                        return [2];
                }
            });
        }); });
        it('single argument', function (math) {
            var hasRan = false;
            math.scope(function () {
                hasRan = true;
            });
            expect(hasRan).toBe(true);
        });
        it('single argument, but not a function throws error', function (math) {
            expect(function () {
                math.scope('asdf');
            }).toThrowError();
        });
        it('2 arguments, first is string', function (math) {
            var hasRan = false;
            math.scope('name', function () {
                hasRan = true;
            });
            expect(hasRan).toBe(true);
        });
        it('2 arguments, but first is not string throws error', function (math) {
            expect(function () {
                math.scope(4, function () { });
            }).toThrowError();
        });
        it('2 arguments, but second is not a function throws error', function (math) {
            expect(function () {
                math.scope('name', 'another name');
            }).toThrowError();
        });
    };
    test_util.describeMathGPU('scope', [gpuTests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var gpuTests = function (it) {
        it('debug mode does not error when no nans', function (math) {
            math.enableDebugMode();
            var a = ndarray_1.Array1D.new([2, -1, 0, 3]);
            var res = math.relu(a);
            test_util.expectArraysClose(res, [2, 0, 0, 3]);
        });
        it('debug mode errors when there are nans, float32', function (math) {
            math.enableDebugMode();
            var a = ndarray_1.Array1D.new([2, NaN]);
            var f = function () { return math.relu(a); };
            expect(f).toThrowError();
        });
        it('debug mode errors when there are nans, int32', function (math) {
            math.enableDebugMode();
            var a = ndarray_1.Array1D.new([2, util.NAN_INT32], 'int32');
            var f = function () { return math.relu(a); };
            expect(f).toThrowError();
        });
        it('debug mode errors when there are nans, bool', function (math) {
            math.enableDebugMode();
            var a = ndarray_1.Array1D.new([1, util.NAN_BOOL], 'bool');
            var f = function () { return math.relu(a); };
            expect(f).toThrowError();
        });
        it('no errors where there are nans, and debug mode is disabled', function (math) {
            var a = ndarray_1.Array1D.new([2, NaN]);
            var res = math.relu(a);
            test_util.expectArraysClose(res, [2, NaN]);
        });
    };
    test_util.describeMathCPU('debug mode', [gpuTests]);
    test_util.describeMathGPU('debug mode', [gpuTests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('debug mode does not error when no nans', function (math) {
            var pixels = new ImageData(2, 2);
            for (var i = 0; i < 8; i++) {
                pixels.data[i] = 100;
            }
            for (var i = 8; i < 16; i++) {
                pixels.data[i] = 250;
            }
            var a = ndarray_1.NDArray.fromPixels(pixels, 4);
            var b = ndarray_1.Scalar.new(20, 'int32');
            var res = math.add(a, b);
            test_util.expectArraysEqual(res, [
                120, 120, 120, 120, 120, 120, 120, 120, 270, 270, 270, 270, 270, 270,
                270, 270
            ]);
        });
    };
    test_util.describeMathGPU('fromPixels + math', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('matmul + relu', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
            var b = ndarray_1.Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 10, 20, 30]);
            var gradients = math.vjp(function () {
                var m = math.matMul(a, b);
                return math.relu(m);
            }, { a: a, b: b }, dy);
            var dedm = math.multiplyStrict(dy, math.step(math.matMul(a, b)));
            expect(gradients.a.shape).toEqual(a.shape);
            test_util.expectArraysClose(gradients.a, math.matMul(dedm, b, matmul_1.MatrixOrientation.REGULAR, matmul_1.MatrixOrientation.TRANSPOSED));
            expect(gradients.b.shape).toEqual(b.shape);
            test_util.expectArraysClose(gradients.b, math.matMul(a, dedm, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.REGULAR));
        });
        it('second order nested gradient vjp & gradients', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Scalar.new(3, 'int32');
            var dy = ndarray_1.Scalar.new(4);
            var gradients = math.vjp(function () {
                return math.gradients(function () { return math.pow(a, b); }, a);
            }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            test_util.expectNumbersClose(gradients.get(), dy.get() * b.get() * (b.get() - 1) * Math.pow(a.get(), b.get() - 2), 1e-1);
        });
        it('second order nested gradient', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Scalar.new(3, 'int32');
            var dy1 = ndarray_1.Scalar.new(3);
            var dy2 = ndarray_1.Scalar.new(4);
            var gradients = math.vjp(function () {
                return math.vjp(function () { return math.pow(a, b); }, a, dy1);
            }, a, dy2);
            expect(gradients.shape).toEqual(a.shape);
            test_util.expectNumbersClose(gradients.get(), dy2.get() * dy1.get() * b.get() * (b.get() - 1) *
                Math.pow(a.get(), b.get() - 2), 1e-1);
        });
    };
    test_util.describeMathCPU('vjp', [tests]);
    test_util.describeMathGPU('vjp', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('matmul + relu', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
            var b = ndarray_1.Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);
            var gradients = math.gradients(function () {
                var m = math.matMul(a, b);
                var y = math.relu(m);
                return math.sum(y);
            }, { a: a, b: b });
            var dedm = math.step(math.matMul(a, b));
            expect(gradients.a.shape).toEqual(a.shape);
            test_util.expectArraysClose(gradients.a, math.matMul(dedm, b, matmul_1.MatrixOrientation.REGULAR, matmul_1.MatrixOrientation.TRANSPOSED), 1e-1);
            expect(gradients.b.shape).toEqual(b.shape);
            test_util.expectArraysClose(gradients.b, math.matMul(a, dedm, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.REGULAR), 1e-1);
        });
        it('second order nested gradient', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var gradients = math.gradients(function () {
                return math.gradients(function () {
                    return math.pow(a, ndarray_1.Scalar.new(3, 'int32'));
                }, a);
            }, a);
            expect(gradients.shape).toEqual(a.shape);
            test_util.expectNumbersClose(gradients.get(), 6 * a.get(), 1e-1);
        });
        it('Throws if y is not a scalar', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
            var b = ndarray_1.Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);
            expect(function () { return math.gradients(function () { return math.matMul(a, b); }, { a: a, b: b }); })
                .toThrowError();
        });
        it('works with reshape', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var exponent = ndarray_1.Array1D.new([2, 2, 2, 2], 'int32');
            var gradients = math.gradients(function () {
                var b = a.flatten();
                var m = math.pow(b, exponent);
                return math.sum(m);
            }, { a: a });
            expect(gradients.a.shape).toEqual([2, 2]);
            test_util.expectArraysClose(gradients.a, [2, 4, 6, 8]);
        });
        it('reshape outside math.gradients() throws error', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var b = a.flatten();
            var exponent = ndarray_1.Array1D.new([2, 2, 2, 2], 'int32');
            var f = function () {
                return math.gradients(function () {
                    var m = math.pow(b, exponent);
                    return math.sum(m);
                }, { a: a, b: b });
            };
            expect(f).toThrowError();
        });
        it('works with asType', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
            var exponent = ndarray_1.Array2D.new([2, 2], [2, 2, 2, 2], 'int32');
            var gradients = math.gradients(function () {
                var b = a.toFloat();
                var m = math.pow(b, exponent);
                return math.sum(m);
            }, { a: a });
            expect(gradients.a.shape).toEqual([2, 2]);
            expect(gradients.a.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.a, [2, 4, 6, 8]);
        });
        it('asType outside of math.gradients() throws error', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
            var b = a.toFloat();
            var exponent = ndarray_1.Array2D.new([2, 2], [2, 2, 2, 2], 'int32');
            var f = function () {
                return math.gradients(function () {
                    var m = math.pow(b, exponent);
                    return math.sum(m);
                }, { a: a });
            };
            expect(f).toThrowError();
        });
    };
    test_util.describeMathCPU('gradients', [tests]);
    test_util.describeMathGPU('gradients', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('matmul + relu', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
            var b = ndarray_1.Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);
            var _a = math.valueAndGradients(function () {
                var m = math.matMul(a, b);
                var y = math.relu(m);
                return math.sum(y);
            }, { a: a, b: b }), value = _a.value, gradients = _a.gradients;
            test_util.expectNumbersClose(value.get(), 10, 1e-1);
            var dedm = math.step(math.matMul(a, b));
            test_util.expectArraysClose(gradients.a, math.matMul(dedm, b, matmul_1.MatrixOrientation.REGULAR, matmul_1.MatrixOrientation.TRANSPOSED), 1e-1);
            test_util.expectArraysClose(gradients.b, math.matMul(a, dedm, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.REGULAR), 1e-1);
        });
        it('Throws is y is not a scalar', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
            var b = ndarray_1.Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);
            expect(function () { return math.valueAndGradients(function () { return math.matMul(a, b); }, { a: a, b: b }); })
                .toThrowError();
        });
        it('matmul + relu + inner scope', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
            var b = ndarray_1.Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);
            var _a = math.valueAndGradients(function () {
                var m = math.matMul(a, b);
                return math.scope(function () {
                    var y = math.relu(m);
                    return math.sum(y);
                });
            }, { a: a, b: b }), value = _a.value, gradients = _a.gradients;
            test_util.expectNumbersClose(value.get(), 10, 1e-1);
            var dedm = math.step(math.matMul(a, b));
            test_util.expectArraysClose(gradients.a, math.matMul(dedm, b, matmul_1.MatrixOrientation.REGULAR, matmul_1.MatrixOrientation.TRANSPOSED), 1e-1);
            test_util.expectArraysClose(gradients.b, math.matMul(a, dedm, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.REGULAR), 1e-1);
        });
    };
    test_util.describeMathCPU('valueAndGradients', [tests]);
    test_util.describeMathGPU('valueAndGradients', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('second order gradients with gradientsScope', function (math) {
            var a = ndarray_1.Scalar.new(2);
            expect(math.getNumArrays()).toBe(1);
            var gradients = math.gradientsScope(function () {
                var der = math.gradients(function () {
                    var result = math.pow(a, ndarray_1.Scalar.new(3, 'int32'));
                    expect(math.getNumArrays()).toBe(3);
                    return result;
                }, a);
                var numArrays = math.getNumArrays();
                expect(numArrays).toBeGreaterThan(3);
                var result = math.gradients(function () { return der; }, a);
                expect(math.getNumArrays()).toBeGreaterThan(numArrays + 1);
                return result;
            });
            expect(math.getNumArrays()).toBe(2);
            expect(gradients.shape).toEqual(a.shape);
            test_util.expectArraysClose(gradients, [2 * 3 * a.get()], 1e-1);
        });
    };
    test_util.describeMathCPU('gradientsScope', [tests]);
    test_util.describeMathGPU('gradientsScope', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Scalar.new(3);
            var b = ndarray_1.Scalar.new(2, 'int32');
            var dy = ndarray_1.Scalar.new(4);
            var vjp = math.vjp(function () {
                return math.customGradient('test', function () {
                    var value = math.pow(a, b);
                    var gradients = function (dy, y) {
                        return { a: function () { return math.multiply(dy, ndarray_1.Scalar.new(3)); } };
                    };
                    return { value: value, gradients: gradients };
                }, { a: a });
            }, a, dy);
            expect(vjp.shape).toEqual(a.shape);
            test_util.expectArraysClose(vjp, [dy.get() * 3]);
        });
        it('second order derivative through customGradient', function (math) {
            var a = ndarray_1.Scalar.new(3);
            var b = ndarray_1.Scalar.new(2, 'int32');
            var dy1 = ndarray_1.Scalar.new(5);
            var dy2 = ndarray_1.Scalar.new(4);
            var vjp = math.vjp(function () {
                return math.vjp(function () {
                    return math.customGradient('test', function () {
                        var value = math.pow(a, b);
                        var gradients = function (dy, y) {
                            return { a: function () { return math.multiply(dy, a); } };
                        };
                        return { value: value, gradients: gradients };
                    }, { a: a });
                }, a, dy1);
            }, a, dy2);
            expect(vjp.shape).toEqual(a.shape);
            test_util.expectArraysClose(vjp, [dy1.get() * dy2.get()]);
        });
    };
    test_util.describeMathCPU('customGradient', [tests]);
    test_util.describeMathGPU('customGradient', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=math_test.js.map