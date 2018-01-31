import { Array2D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface MultinomialNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            probs: Array2D;
        };
        args: {
            numSamples: number;
            seed: number;
        };
    };
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => {
        probs: () => Array2D;
    };
}
