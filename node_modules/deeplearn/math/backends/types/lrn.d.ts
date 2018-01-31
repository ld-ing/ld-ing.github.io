import { Array4D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface LRN4DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array4D;
        };
        args: {
            radius: number;
            bias: number;
            alpha: number;
            beta: number;
            normRegion: 'acrossChannels' | 'withinChannel';
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        x: () => Array4D;
    };
}
