import { Array1D, Array2D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface Pad1DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array1D;
        };
        args: {
            paddings: [number, number];
            constantValue: number;
        };
    };
    output: Array1D;
    gradient: (dy: Array1D, y: Array1D) => {
        x: () => Array1D;
    };
}
export interface Pad2DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array2D;
        };
        args: {
            paddings: [[number, number], [number, number]];
            constantValue: number;
        };
    };
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => {
        x: () => Array2D;
    };
}
