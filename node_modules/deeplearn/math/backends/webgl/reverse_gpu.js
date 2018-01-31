"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ReverseProgram = (function () {
    function ReverseProgram(xShape, axis) {
        this.variableNames = ['x'];
        this.outputShape = xShape;
        var getRevVar = function (i) {
            if (axis.indexOf(i) !== -1 && xShape[i] !== 1) {
                return xShape[i] + " - coords[" + i + "] - 1";
            }
            return "coords[" + i + "]";
        };
        var b = getRevVar(0);
        var r = getRevVar(1);
        var c = getRevVar(2);
        var d = getRevVar(3);
        this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        float val = getX(" + b + ", " + r + ", " + c + ", " + d + ");\n        setOutput(val);\n      }\n    ";
    }
    return ReverseProgram;
}());
exports.ReverseProgram = ReverseProgram;
//# sourceMappingURL=reverse_gpu.js.map