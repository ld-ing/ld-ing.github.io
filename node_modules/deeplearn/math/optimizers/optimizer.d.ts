import { Node, VariableNode } from '../../graph/graph';
import { SessionRuntime } from '../../graph/session';
import { SummedTensorArrayMap, TensorArrayMap } from '../../graph/tensor_array_map';
import { NDArrayMath } from '../../math/math';
import { Scalar, Variable } from '../../math/ndarray';
import { NamedArrayMap } from '../types';
export declare abstract class Optimizer {
    protected learningRate: number;
    protected variableNodes: VariableNode[];
    protected specifiedVariableNodes: VariableNode[] | null;
    constructor(learningRate: number, specifiedVariableList?: Node[]);
    minimize(f: () => Scalar, returnCost?: boolean, varList?: Variable[]): Scalar | null;
    computeGradients(f: () => Scalar, varList?: Variable[]): {
        value: Scalar;
        gradients: NamedArrayMap;
    };
    abstract applyGradients(variableGradients: NamedArrayMap): void;
    beforeBatch(math: NDArrayMath, batchSize: number, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    afterExample(math: NDArrayMath, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    abstract afterBatch(math: NDArrayMath, batchSize: number, runtime: SessionRuntime, activationArrayMap: TensorArrayMap, gradientArrayMap: SummedTensorArrayMap): void;
    dispose(): void;
    protected variableGradients: TensorArrayMap;
    protected prevBatchSize: number;
    protected one: Scalar;
    protected cGraph: Scalar;
}
