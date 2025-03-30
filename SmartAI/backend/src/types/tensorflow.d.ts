declare module '@tensorflow/tfjs-node' {
  export * from '@tensorflow/tfjs';
  
  export interface Tensor {
    data(): Promise<Float32Array>;
  }
  
  export interface LayersModel {
    predict(input: Tensor): Tensor;
    save(path: string): Promise<void>;
    compile(config: any): void;
  }
  
  export interface Sequential {
    layers: Layer[];
  }
  
  export interface Layer {
    units: number;
    activation?: string;
    inputShape?: number[];
    rate?: number;
  }
  
  export const sequential: (config: { layers: Layer[] }) => LayersModel;
  export const layers: {
    dense: (config: Layer) => Layer;
    lstm: (config: Layer) => Layer;
    dropout: (config: { rate: number }) => Layer;
    layerNormalization: () => Layer;
  };
  export const train: {
    adam: (learningRate: number) => any;
  };
  export const loadLayersModel: (path: string) => Promise<LayersModel>;
  export const tensor2d: (data: number[][]) => Tensor;
  export const tensor3d: (data: number[][][]) => Tensor;
} 