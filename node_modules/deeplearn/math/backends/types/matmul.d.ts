import { Array2D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface MatMulNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            a: Array2D;
            b: Array2D;
        };
        args: {
            aOrientation: MatrixOrientation;
            bOrientation: MatrixOrientation;
        };
    };
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => {
        a: () => Array2D;
        b: () => Array2D;
    };
}
export declare enum MatrixOrientation {
    REGULAR = 0,
    TRANSPOSED = 1,
}
