import { NDArray, Variable } from './ndarray';
export declare enum DType {
    float32 = "float32",
    int32 = "int32",
    bool = "bool",
}
export interface ShapeMap {
    R0: number[];
    R1: [number];
    R2: [number, number];
    R3: [number, number, number];
    R4: [number, number, number, number];
}
export interface DataTypeMap {
    float32: Float32Array;
    int32: Int32Array;
    bool: Uint8Array;
}
export declare type DataType = keyof DataTypeMap;
export declare type TypedArray = DataTypeMap[DataType];
export declare enum Rank {
    R0 = "R0",
    R1 = "R1",
    R2 = "R2",
    R3 = "R3",
    R4 = "R4",
}
export declare type FlatVector = boolean[] | number[] | TypedArray;
export declare type RegularArray<T> = T[] | T[][] | T[][][] | T[][][][];
export declare type ArrayData<D extends DataType> = DataTypeMap[D] | RegularArray<number> | RegularArray<boolean>;
export declare type NamedArrayMap = {
    [name: string]: NDArray;
};
export declare type NamedVariableMap = {
    [name: string]: Variable;
};
export declare function upcastType(typeA: DataType, typeB: DataType): DataType;
export declare function sumOutType(type: DataType): "float32" | "int32" | "bool";
