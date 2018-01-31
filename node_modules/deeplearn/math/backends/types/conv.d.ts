import { Conv2DInfo } from '../../conv_util';
import { Array1D, Array4D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface Conv2DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array4D;
            filter: Array4D;
            bias?: Array1D;
        };
        args: {
            convInfo: Conv2DInfo;
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        x: () => Array4D;
        filter: () => Array4D;
        bias?: () => Array1D;
    };
}
export interface Conv2DDerInputNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            dy: Array4D;
            filter: Array4D;
        };
        args: {
            convInfo: Conv2DInfo;
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        dy: () => Array4D;
        filter: () => Array4D;
    };
}
export interface Conv2DDerFilterNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array4D;
            dy: Array4D;
        };
        args: {
            convInfo: Conv2DInfo;
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        x: () => Array4D;
        dy: () => Array4D;
    };
}
export interface Conv2DDerBiasNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            dy: Array4D;
        };
    };
    output: Array1D;
    gradient: (dy: Array1D, y: Array1D) => {
        dy: () => Array4D;
    };
}
export interface DepthwiseConv2DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array4D;
            filter: Array4D;
        };
        args: {
            convInfo: Conv2DInfo;
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        x: () => Array4D;
        filter: () => Array4D;
    };
}
