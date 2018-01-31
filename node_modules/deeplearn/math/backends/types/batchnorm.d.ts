import { Array1D, Array2D, Array3D, Array4D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface BatchNorm4DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array4D;
            mean: Array4D | Array1D;
            variance: Array4D | Array1D;
            scale?: Array4D | Array1D;
            offset?: Array4D | Array1D;
        };
        args: {
            varianceEpsilon: number;
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        x: () => Array4D;
        mean: () => Array4D | Array1D;
        variance: () => Array4D | Array1D;
        scale?: () => Array4D | Array1D;
        offset?: () => Array4D | Array1D;
    };
}
export interface BatchNorm3DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array3D;
            mean: Array3D | Array1D;
            variance: Array3D | Array1D;
            scale?: Array3D | Array1D;
            offset?: Array3D | Array1D;
        };
        args: {
            varianceEpsilon: number;
        };
    };
    output: Array3D;
    gradient: (dy: Array3D, y: Array3D) => {
        x: () => Array3D;
        mean: () => Array3D | Array1D;
        variance: () => Array3D | Array1D;
        scale?: () => Array3D | Array1D;
        offset?: () => Array3D | Array1D;
    };
}
export interface BatchNorm2DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array2D;
            mean: Array2D | Array1D;
            variance: Array2D | Array1D;
            scale?: Array2D | Array1D;
            offset?: Array2D | Array1D;
        };
        args: {
            varianceEpsilon: number;
        };
    };
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => {
        x: () => Array2D;
        mean: () => Array2D | Array1D;
        variance: () => Array2D | Array1D;
        scale?: () => Array2D | Array1D;
        offset?: () => Array2D | Array1D;
    };
}
