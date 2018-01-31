import { NDArrayMath } from '../../math/math';
import { Optimizer } from '../../math/optimizers/optimizer';
import { NamedVariableMap } from '../../math/types';
import { Node } from '../graph';
import { SessionRuntime } from '../session';
import { SummedTensorArrayMap, TensorArrayMap } from '../tensor_array_map';
export declare class AdagradOptimizer extends Optimizer {
    protected learningRate: number;
    constructor(learningRate: number, specifiedVariableList?: Node[]);
    applyGradients(variableGradients: NamedVariableMap): void;
    beforeBatch(math: NDArrayMath, batchSize: number, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    afterBatch(math: NDArrayMath, batchSize: number, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    dispose(): void;
    private accumulatedSquaredGradients;
    private eps;
}
