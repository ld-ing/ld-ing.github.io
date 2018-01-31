import { Array1D, Array2D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface OneHotNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            indices: Array1D;
        };
        args: {
            depth: number;
            onValue: number;
            offValue: number;
        };
    };
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => {
        indices: () => Array1D;
    };
}
