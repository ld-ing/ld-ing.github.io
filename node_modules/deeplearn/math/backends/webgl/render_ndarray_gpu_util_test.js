"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_context_1 = require("./gpgpu_context");
var gpgpu_util = require("./gpgpu_util");
var render_ndarray_gpu_util = require("./render_ndarray_gpu_util");
function uploadRenderRGBDownload(source, sourceShape) {
    var canvas = document.createElement('canvas');
    canvas.width = sourceShape[0];
    canvas.height = sourceShape[1];
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    var program = render_ndarray_gpu_util.getRenderRGBShader(gpgpu, sourceShape[1]);
    var sourceTexShapeRC = [sourceShape[0], sourceShape[1] * sourceShape[2]];
    var sourceTex = gpgpu.createMatrixTexture(sourceTexShapeRC[0], sourceTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(sourceTex, sourceTexShapeRC[0], sourceTexShapeRC[1], source);
    var resultTex = gpgpu_util.createColorMatrixTexture(gpgpu.gl, sourceShape[0], sourceShape[1]);
    gpgpu.setOutputMatrixTexture(resultTex, sourceShape[0], sourceShape[1]);
    render_ndarray_gpu_util.renderToFramebuffer(gpgpu, program, sourceTex);
    var result = new Float32Array(sourceShape[0] * sourceShape[1] * 4);
    gpgpu.gl.readPixels(0, 0, sourceShape[1], sourceShape[0], gpgpu.gl.RGBA, gpgpu.gl.FLOAT, result);
    return result;
}
describe('render_gpu', function () {
    it('Packs a 1x1x3 vector to a 1x1 color texture', function () {
        var source = new Float32Array([1, 2, 3]);
        var result = uploadRenderRGBDownload(source, [1, 1, 3]);
        expect(result).toEqual(new Float32Array([1, 2, 3, 1]));
    });
    it('Packs a 2x2x3 vector to a 2x2 color texture, mirrored vertically', function () {
        var source = new Float32Array([1, 2, 3, 30, 20, 10, 2, 3, 4, 40, 30, 20]);
        var result = uploadRenderRGBDownload(source, [2, 2, 3]);
        expect(result).toEqual(new Float32Array([2, 3, 4, 1, 40, 30, 20, 1, 1, 2, 3, 1, 30, 20, 10, 1]));
    });
});
//# sourceMappingURL=render_ndarray_gpu_util_test.js.map