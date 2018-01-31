import { NDArrayMath } from '../../math/math';
import { SGDOptimizer } from '../../math/optimizers/sgd_optimizer';
import { NamedVariableMap } from '../../math/types';
import { Node } from '../graph';
import { SessionRuntime } from '../session';
import { SummedTensorArrayMap, TensorArrayMap } from '../tensor_array_map';
export declare class MomentumOptimizer extends SGDOptimizer {
    protected learningRate: number;
    private momentum;
    constructor(learningRate: number, momentum: number, specifiedVariableList?: Node[]);
    applyGradients(variableGradients: NamedVariableMap): void;
    beforeBatch(math: NDArrayMath, batchSize: number, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    afterBatch(math: NDArrayMath, batchSize: number, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    dispose(): void;
    setMomentum(momentum: number): void;
    private variableVelocities;
    private m;
}
