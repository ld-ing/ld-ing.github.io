import { Array2D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface ConcatNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            a: Array2D;
            b: Array2D;
        };
    };
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => {
        a: () => Array2D;
        b: () => Array2D;
    };
}
