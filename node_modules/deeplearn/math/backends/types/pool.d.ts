import { Conv2DInfo } from '../../conv_util';
import { Array4D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface PoolNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array4D;
        };
        args: {
            convInfo: Conv2DInfo;
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        x: () => Array4D;
    };
}
export interface PoolBackpropNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            dy: Array4D;
            x: Array4D;
        };
        args: {
            convInfo: Conv2DInfo;
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        dy: () => Array4D;
        x: () => Array4D;
    };
}
