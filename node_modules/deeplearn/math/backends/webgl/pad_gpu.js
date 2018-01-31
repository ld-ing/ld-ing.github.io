"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Pad1DProgram = (function () {
    function Pad1DProgram(xShape, paddings, constantValue) {
        this.variableNames = ['x'];
        var leftPadding = paddings[0];
        var rightPadding = paddings[1];
        this.outputShape = [leftPadding + xShape[0] + rightPadding];
        this.rank = 1;
        this.userCode = "\n      void main() {\n        int resRC = getOutputCoords();\n        if (resRC < " + leftPadding + " || resRC >= " + leftPadding + " + " + xShape[0] + ") {\n          setOutput(float(" + constantValue + "));\n        } else {\n          setOutput(getX(resRC - " + leftPadding + "));\n        }\n      }\n    ";
    }
    return Pad1DProgram;
}());
exports.Pad1DProgram = Pad1DProgram;
var Pad2DProgram = (function () {
    function Pad2DProgram(xShape, paddings, constantValue) {
        this.variableNames = ['x'];
        var topPadding = paddings[0][0];
        var bottomPadding = paddings[0][1];
        var leftPadding = paddings[1][0];
        var rightPadding = paddings[1][1];
        this.outputShape = [
            topPadding + xShape[0] + bottomPadding,
            leftPadding + xShape[1] + rightPadding
        ];
        this.rank = 2;
        var sourceCoords = "resRC.x - " + topPadding + ", resRC.y - " + leftPadding;
        this.userCode = "\n      void main() {\n        ivec2 resRC = getOutputCoords();\n        int topShape = " + topPadding + " + " + xShape[0] + ";\n        int leftShape = " + leftPadding + " + " + xShape[1] + ";\n        if (resRC.x < " + topPadding + " || resRC.x >= topShape ||\n            resRC.y < " + leftPadding + " || resRC.y >= leftShape) {\n          setOutput(float(" + constantValue + "));\n        } else {\n          setOutput(getX(" + sourceCoords + "));\n        }\n      }\n    ";
    }
    return Pad2DProgram;
}());
exports.Pad2DProgram = Pad2DProgram;
//# sourceMappingURL=pad_gpu.js.map