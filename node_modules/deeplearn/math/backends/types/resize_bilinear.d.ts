import { Array4D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface ResizeBilinearNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array4D;
        };
        args: {
            newHeight: number;
            newWidth: number;
            alignCorners: boolean;
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        x: () => Array4D;
    };
}
