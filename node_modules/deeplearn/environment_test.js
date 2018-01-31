"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var device_util = require("./device_util");
var environment_1 = require("./environment");
var backend_cpu_1 = require("./math/backends/backend_cpu");
var backend_webgl_1 = require("./math/backends/backend_webgl");
describe('disjoint query timer enabled', function () {
    afterEach(function () {
        environment_1.ENV.reset();
    });
    it('no webgl', function () {
        environment_1.ENV.setFeatures({ 'WEBGL_VERSION': 0 });
        expect(environment_1.ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')).toBe(false);
    });
    it('webgl 1', function () {
        var features = { 'WEBGL_VERSION': 1 };
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) {
                if (context === 'webgl' || context === 'experimental-webgl') {
                    return {
                        getExtension: function (extensionName) {
                            if (extensionName === 'EXT_disjoint_timer_query') {
                                return {};
                            }
                            else if (extensionName === 'WEBGL_lose_context') {
                                return { loseContext: function () { } };
                            }
                            return null;
                        }
                    };
                }
                return null;
            }
        });
        environment_1.ENV.setFeatures(features);
        expect(environment_1.ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')).toBe(true);
    });
    it('webgl 2', function () {
        var features = { 'WEBGL_VERSION': 2 };
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) {
                if (context === 'webgl2') {
                    return {
                        getExtension: function (extensionName) {
                            if (extensionName === 'EXT_disjoint_timer_query_webgl2') {
                                return {};
                            }
                            else if (extensionName === 'WEBGL_lose_context') {
                                return { loseContext: function () { } };
                            }
                            return null;
                        }
                    };
                }
                return null;
            }
        });
        environment_1.ENV.setFeatures(features);
        expect(environment_1.ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')).toBe(true);
    });
});
describe('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', function () {
    afterEach(function () {
        environment_1.ENV.reset();
    });
    it('disjoint query timer disabled', function () {
        var features = { 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED': false };
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(false);
    });
    it('disjoint query timer enabled, mobile', function () {
        var features = { 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED': true };
        spyOn(device_util, 'isMobile').and.returnValue(true);
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(false);
    });
    it('disjoint query timer enabled, not mobile', function () {
        var features = { 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED': true };
        spyOn(device_util, 'isMobile').and.returnValue(false);
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')).toBe(true);
    });
});
describe('WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED', function () {
    afterEach(function () {
        environment_1.ENV.reset();
    });
    beforeEach(function () {
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) {
                if (context === 'webgl2') {
                    return {
                        getExtension: function (extensionName) {
                            if (extensionName === 'WEBGL_get_buffer_sub_data_async') {
                                return {};
                            }
                            else if (extensionName === 'WEBGL_lose_context') {
                                return { loseContext: function () { } };
                            }
                            return null;
                        }
                    };
                }
                return null;
            }
        });
    });
    it('WebGL 2 enabled', function () {
        var features = { 'WEBGL_VERSION': 2 };
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED'))
            .toBe(true);
    });
    it('WebGL 1 disabled', function () {
        var features = { 'WEBGL_VERSION': 1 };
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED'))
            .toBe(false);
    });
});
describe('WebGL version', function () {
    afterEach(function () {
        environment_1.ENV.reset();
    });
    it('webgl 1', function () {
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) {
                if (context === 'webgl') {
                    return {
                        getExtension: function (a) {
                            return { loseContext: function () { } };
                        }
                    };
                }
                return null;
            }
        });
        var env = new environment_1.Environment();
        expect(env.get('WEBGL_VERSION')).toBe(1);
    });
    it('webgl 2', function () {
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) {
                if (context === 'webgl2') {
                    return {
                        getExtension: function (a) {
                            return { loseContext: function () { } };
                        }
                    };
                }
                return null;
            }
        });
        var env = new environment_1.Environment();
        expect(env.get('WEBGL_VERSION')).toBe(2);
    });
    it('no webgl', function () {
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) { return null; }
        });
        var env = new environment_1.Environment();
        expect(env.get('WEBGL_VERSION')).toBe(0);
    });
});
describe('Backend', function () {
    afterEach(function () {
        environment_1.ENV.reset();
    });
    it('default ENV has cpu and webgl, and webgl is the best available', function () {
        expect(environment_1.ENV.getBackend('webgl') != null).toBe(true);
        expect(environment_1.ENV.getBackend('cpu') != null).toBe(true);
        expect(environment_1.ENV.getBestBackendType()).toBe('webgl');
    });
    it('custom webgl registration', function () {
        var features = { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 };
        environment_1.ENV.setFeatures(features);
        var backend;
        environment_1.ENV.addCustomBackend('webgl', function () {
            backend = new backend_webgl_1.MathBackendWebGL();
            return backend;
        });
        expect(environment_1.ENV.getBackend('webgl')).toBe(backend);
        expect(environment_1.ENV.math).not.toBeNull();
    });
    it('double registration fails', function () {
        environment_1.ENV.setFeatures({ 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 });
        environment_1.ENV.addCustomBackend('webgl', function () { return new backend_webgl_1.MathBackendWebGL(); });
        expect(function () { return environment_1.ENV.addCustomBackend('webgl', function () { return new backend_webgl_1.MathBackendWebGL(); }); })
            .toThrowError();
    });
    it('webgl not supported, falls back to cpu', function () {
        environment_1.ENV.setFeatures({ 'WEBGL_VERSION': 0 });
        environment_1.ENV.addCustomBackend('cpu', function () { return new backend_cpu_1.MathBackendCPU(); });
        var success = environment_1.ENV.addCustomBackend('webgl', function () { return new backend_webgl_1.MathBackendWebGL(); });
        expect(success).toBe(false);
        expect(environment_1.ENV.getBackend('webgl') == null).toBe(true);
        expect(environment_1.ENV.getBestBackendType()).toBe('cpu');
    });
});
//# sourceMappingURL=environment_test.js.map