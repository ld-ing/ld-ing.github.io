import { Node } from '../../graph/graph';
import { SessionRuntime } from '../../graph/session';
import { SummedTensorArrayMap, TensorArrayMap } from '../../graph/tensor_array_map';
import { NDArrayMath } from '../../math/math';
import { NamedArrayMap } from '../types';
import { Optimizer } from './optimizer';
export declare class SGDOptimizer extends Optimizer {
    protected learningRate: number;
    private c;
    constructor(learningRate: number, specifiedVariableList?: Node[]);
    applyGradients(variableGradients: NamedArrayMap): void;
    afterBatch(math: NDArrayMath, batchSize: number, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    dispose(): void;
    setLearningRate(learningRate: number): void;
}
