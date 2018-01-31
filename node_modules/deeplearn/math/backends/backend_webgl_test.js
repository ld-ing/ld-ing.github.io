"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../../test_util");
var backend_webgl_1 = require("./backend_webgl");
var tests = function () {
    it('delayed storage, reading', function () {
        var delayedStorage = true;
        var backend = new backend_webgl_1.MathBackendWebGL(null, delayedStorage);
        var texManager = backend.getTextureManager();
        backend.register(0, [3], 'float32');
        backend.write(0, new Float32Array([1, 2, 3]));
        expect(texManager.getNumUsedTextures()).toBe(0);
        backend.getTexture(0);
        expect(texManager.getNumUsedTextures()).toBe(1);
        test_util.expectArraysClose(backend.readSync(0), new Float32Array([1, 2, 3]));
        expect(texManager.getNumUsedTextures()).toBe(0);
        backend.getTexture(0);
        expect(texManager.getNumUsedTextures()).toBe(1);
        backend.disposeData(0);
        expect(texManager.getNumUsedTextures()).toBe(0);
    });
    it('delayed storage, overwriting', function () {
        var delayedStorage = true;
        var backend = new backend_webgl_1.MathBackendWebGL(null, delayedStorage);
        var texManager = backend.getTextureManager();
        backend.register(0, [3], 'float32');
        backend.write(0, new Float32Array([1, 2, 3]));
        backend.getTexture(0);
        expect(texManager.getNumUsedTextures()).toBe(1);
        backend.write(0, new Float32Array([4, 5, 6]));
        expect(texManager.getNumUsedTextures()).toBe(0);
        test_util.expectArraysClose(backend.readSync(0), new Float32Array([4, 5, 6]));
        backend.getTexture(0);
        expect(texManager.getNumUsedTextures()).toBe(1);
        test_util.expectArraysClose(backend.readSync(0), new Float32Array([4, 5, 6]));
        expect(texManager.getNumUsedTextures()).toBe(0);
    });
    it('immediate storage reading', function () {
        var delayedStorage = false;
        var backend = new backend_webgl_1.MathBackendWebGL(null, delayedStorage);
        var texManager = backend.getTextureManager();
        backend.register(0, [3], 'float32');
        backend.write(0, new Float32Array([1, 2, 3]));
        expect(texManager.getNumUsedTextures()).toBe(1);
        test_util.expectArraysClose(backend.readSync(0), new Float32Array([1, 2, 3]));
        expect(texManager.getNumUsedTextures()).toBe(1);
        backend.disposeData(0);
        expect(texManager.getNumUsedTextures()).toBe(0);
    });
    it('immediate storage overwriting', function () {
        var delayedStorage = false;
        var backend = new backend_webgl_1.MathBackendWebGL(null, delayedStorage);
        var texManager = backend.getTextureManager();
        backend.register(0, [3], 'float32');
        backend.write(0, new Float32Array([1, 2, 3]));
        expect(texManager.getNumUsedTextures()).toBe(1);
        backend.write(0, new Float32Array([4, 5, 6]));
        expect(texManager.getNumUsedTextures()).toBe(1);
        test_util.expectArraysClose(backend.readSync(0), new Float32Array([4, 5, 6]));
        expect(texManager.getNumUsedTextures()).toBe(1);
        backend.disposeData(0);
        expect(texManager.getNumUsedTextures()).toBe(0);
    });
    it('disposal of backend disposes all textures', function () {
        var delayedStorage = false;
        var backend = new backend_webgl_1.MathBackendWebGL(null, delayedStorage);
        var texManager = backend.getTextureManager();
        backend.register(0, [3], 'float32');
        backend.write(0, new Float32Array([1, 2, 3]));
        backend.register(1, [3], 'float32');
        backend.write(1, new Float32Array([4, 5, 6]));
        expect(texManager.getNumUsedTextures()).toBe(2);
        backend.dispose();
        expect(texManager.getNumUsedTextures()).toBe(0);
    });
};
test_util.describeCustom('backend_webgl', tests, [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=backend_webgl_test.js.map