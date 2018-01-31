import { MatrixOrientation } from './backends/types/matmul';
import { Array1D, Array2D, Scalar } from './ndarray';
export declare class Ops {
    static matMul(a: Array2D, b: Array2D, aOrientation?: MatrixOrientation, bOrientation?: MatrixOrientation): Array2D;
    static vectorTimesMatrix(v: Array1D, matrix: Array2D): Array1D;
    static matrixTimesVector(matrix: Array2D, v: Array1D): Array1D;
    static dotProduct(v1: Array1D, v2: Array1D): Scalar;
    static outerProduct(v1: Array1D, v2: Array1D): Array2D;
}
