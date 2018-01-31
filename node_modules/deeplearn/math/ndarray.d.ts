import { MatrixOrientation } from './backends/types/matmul';
import { RandNormalDataTypes } from './rand';
import { DataType, DataTypeMap, Rank, ShapeMap, TypedArray } from './types';
export interface NDArrayData {
    dataId?: number;
    values?: TypedArray;
}
export declare class NDArray<R extends Rank = Rank> {
    private static nextId;
    private static nextDataId;
    id: number;
    dataId: number;
    shape: ShapeMap[R];
    size: number;
    dtype: DataType;
    rankType: R;
    strides: number[];
    protected constructor(shape: ShapeMap[R], dtype: DataType, values?: TypedArray, dataId?: number);
    static ones<R extends Rank>(shape: ShapeMap[R], dtype?: DataType): NDArray<R>;
    static zeros<R extends Rank>(shape: ShapeMap[R], dtype?: DataType): NDArray<R>;
    static onesLike<T extends NDArray>(x: T): T;
    static zerosLike<T extends NDArray>(x: T): T;
    static like<T extends NDArray>(x: T): T;
    static make<T extends NDArray<R>, D extends DataType = 'float32', R extends Rank = Rank>(shape: ShapeMap[R], data: NDArrayData, dtype?: D): T;
    static fromPixels(pixels: ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, numChannels?: number): Array3D;
    static rand<R extends Rank>(shape: ShapeMap[R], randFunction: () => number, dtype?: DataType): NDArray<R>;
    static randNormal<R extends Rank>(shape: ShapeMap[R], mean?: number, stdDev?: number, dtype?: keyof RandNormalDataTypes, seed?: number): NDArray<R>;
    static randTruncatedNormal<R extends Rank>(shape: ShapeMap[R], mean?: number, stdDev?: number, dtype?: keyof RandNormalDataTypes, seed?: number): NDArray<R>;
    static randUniform<R extends Rank>(shape: ShapeMap[R], a: number, b: number, dtype?: DataType): NDArray<R>;
    squeeze<T extends NDArray>(axis?: number[]): T;
    flatten(): Array1D;
    asScalar(): Scalar;
    as1D(): Array1D;
    as2D(rows: number, columns: number): Array2D;
    as3D(rows: number, columns: number, depth: number): Array3D;
    as4D(rows: number, columns: number, depth: number, depth2: number): Array4D;
    asType<T extends this>(this: T, dtype: DataType): T;
    readonly rank: number;
    get(...locs: number[]): number;
    set(value: number, ...locs: number[]): void;
    val(...locs: number[]): Promise<number>;
    locToIndex(locs: number[]): number;
    indexToLoc(index: number): number[];
    fill(value: number): void;
    getValues(): TypedArray;
    getValuesAsync(): Promise<TypedArray>;
    data(): Promise<TypedArray>;
    dataSync(): TypedArray;
    dispose(): void;
    private isDisposed;
    protected throwIfDisposed(): void;
    toFloat<T extends this>(this: T): T;
    toInt(): this;
    toBool(): this;
    reshape<R2 extends Rank>(newShape: ShapeMap[R2]): NDArray<R2>;
    reshapeAs<T extends NDArray>(x: T): T;
    tile<T extends this>(this: T, reps: number[]): T;
    gather<T extends this>(this: T, indices: Array1D, axis?: number): T;
    matMul(b: Array2D, aOrientation?: MatrixOrientation, bOrientation?: MatrixOrientation): Array2D;
    slice(begin: ShapeMap[R], size: ShapeMap[R]): NDArray<R>;
    reverse(axis: number | number[]): NDArray<R>;
    concat(x: NDArray<R>, axis: number): NDArray<R>;
    batchNormalization(mean: NDArray<R> | Array1D, variance: NDArray<R> | Array1D, varianceEpsilon?: number, scale?: NDArray<R> | Array1D, offset?: NDArray<R> | Array1D): NDArray<R>;
    clone(): NDArray<R>;
    logSumExp<T extends NDArray>(axis?: number | number[], keepDims?: boolean): T;
    sum<T extends NDArray>(axis?: number | number[], keepDims?: boolean): T;
    mean<T extends NDArray>(axis?: number | number[], keepDims?: boolean): T;
    min<T extends NDArray>(axis?: number | number[], keepDims?: boolean): T;
    max<T extends NDArray>(axis?: number | number[], keepDims?: boolean): T;
    argMin<T extends NDArray>(axis?: number): T;
    argMax<T extends NDArray>(axis?: number): T;
    argMaxEquals(x: NDArray): Scalar;
    add<T extends NDArray>(x: NDArray): T;
    addStrict<T extends this>(this: T, x: T): T;
    sub<T extends NDArray>(x: NDArray): T;
    subStrict<T extends this>(this: T, x: T): T;
    pow<T extends NDArray>(exp: NDArray): T;
    powStrict(exp: NDArray): NDArray<R>;
    mul<T extends NDArray>(x: NDArray): T;
    mulStrict<T extends this>(this: T, x: T): T;
    div<T extends NDArray>(x: NDArray): T;
    divStrict<T extends this>(this: T, x: T): T;
    minimum<T extends NDArray>(x: NDArray): T;
    minimumStrict<T extends this>(this: T, x: T): T;
    maximum<T extends NDArray>(x: NDArray): T;
    maximumStrict<T extends this>(this: T, x: T): T;
    transpose(perm?: number[]): NDArray<R>;
    notEqual<T extends NDArray>(x: NDArray): T;
    notEqualStrict<T extends this>(this: T, x: T): T;
    less<T extends NDArray>(x: NDArray): T;
    lessStrict<T extends this>(this: T, x: T): T;
    equal<T extends NDArray>(x: NDArray): T;
    equalStrict<T extends this>(this: T, x: T): T;
    lessEqual<T extends NDArray>(x: NDArray): T;
    lessEqualStrict<T extends this>(this: T, x: T): T;
    greater<T extends NDArray>(x: NDArray): T;
    greaterStrict<T extends this>(this: T, x: T): T;
    greaterEqual<T extends NDArray>(x: NDArray): T;
    greaterEqualStrict<T extends this>(this: T, x: T): T;
    logicalAnd(x: NDArray): NDArray;
    logicalOr(x: NDArray): NDArray;
    where(condition: NDArray, x: NDArray): NDArray;
    neg(): NDArray<R>;
    ceil(): NDArray<R>;
    floor(): NDArray<R>;
    exp(): NDArray<R>;
    log(): NDArray<R>;
    sqrt(): NDArray<R>;
    square(): NDArray<R>;
    abs(): NDArray<R>;
    clip(min: number, max: number): NDArray<R>;
    relu(): NDArray<R>;
    elu(): NDArray<R>;
    selu(): NDArray<R>;
    leakyRelu(alpha?: number): NDArray<R>;
    prelu(alpha: NDArray<R>): NDArray<R>;
    sigmoid(): NDArray<R>;
    sin(): NDArray<R>;
    cos(): NDArray<R>;
    tan(): NDArray<R>;
    asin(): NDArray<R>;
    acos(): NDArray<R>;
    atan(): NDArray<R>;
    sinh(): NDArray<R>;
    cosh(): NDArray<R>;
    tanh(): NDArray<R>;
    step(alpha?: number): NDArray<R>;
    softmax<T extends this>(this: T, dim?: number): T;
    resizeBilinear<T extends Array3D | Array4D>(this: T, newShape2D: [number, number], alignCorners?: boolean): T;
    conv1d<T extends Array2D | Array3D>(this: T, filter: Array3D, bias: Array1D | null, stride: number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    conv2d<T extends Array3D | Array4D>(this: T, filter: Array4D, bias: Array1D | null, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    conv2dTranspose<T extends Array3D | Array4D>(this: T, filter: Array4D, outputShape: [number, number, number, number] | [number, number, number], strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    depthwiseConv2D<T extends Array3D | Array4D>(this: T, filter: Array4D, strides: [number, number] | number, pad: 'valid' | 'same' | number, rates?: [number, number] | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    avgPool<T extends Array3D | Array4D>(this: T, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    maxPool<T extends Array3D | Array4D>(this: T, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    minPool<T extends Array3D | Array4D>(this: T, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
}
export declare class Scalar extends NDArray<Rank.R0> {
    static new(value: number | boolean, dtype?: DataType): Scalar;
}
export declare class Array1D extends NDArray<Rank.R1> {
    static new<D extends DataType = 'float32'>(values: DataTypeMap[D] | number[] | boolean[], dtype?: D): Array1D;
}
export declare class Array2D extends NDArray<Rank.R2> {
    static new<D extends DataType = 'float32'>(shape: [number, number], values: DataTypeMap[D] | number[] | number[][] | boolean[] | boolean[][], dtype?: D): Array2D;
}
export declare class Array3D extends NDArray<Rank.R3> {
    static new<D extends DataType = 'float32'>(shape: [number, number, number], values: DataTypeMap[D] | number[] | number[][][] | boolean[] | boolean[][][], dtype?: D): Array3D;
}
export declare class Array4D extends NDArray<Rank.R4> {
    static new<D extends DataType = 'float32'>(shape: [number, number, number, number], values: DataTypeMap[D] | number[] | number[][][][] | boolean[] | boolean[][][][], dtype?: D): Array4D;
}
export declare class Variable<R extends Rank = Rank> extends NDArray<R> {
    trainable: boolean;
    private static nextVarId;
    name: string;
    private constructor();
    static variable<R extends Rank>(initialValue: NDArray<R>, trainable?: boolean, name?: string, dtype?: DataType): Variable<R>;
    assign(newValue: NDArray<R>): void;
}
declare const variable: typeof Variable.variable;
export { variable };
