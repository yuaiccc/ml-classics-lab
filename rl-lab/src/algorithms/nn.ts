// 极简单隐层 MLP（in → hidden[tanh] → out[linear]）+ 手写反向传播。
// 给 GAN / 扩散 / 等需要小神经网络的浏览器端算法复用，避免重复造轮子。
export interface MLP {
  W1: number[][]; // [H][in]
  b1: number[];
  W2: number[][]; // [out][H]
  b2: number[];
  inDim: number;
  hidden: number;
  outDim: number;
}

export function makeMLP(inDim: number, hidden: number, outDim: number, rand: () => number): MLP {
  const r = () => (rand() * 2 - 1) * 0.6;
  return {
    inDim,
    hidden,
    outDim,
    W1: Array.from({ length: hidden }, () => Array.from({ length: inDim }, r)),
    b1: new Array(hidden).fill(0),
    W2: Array.from({ length: outDim }, () => Array.from({ length: hidden }, r)),
    b2: new Array(outDim).fill(0),
  };
}

export interface Forward {
  h: number[];
  out: number[];
}

export function forward(net: MLP, x: number[]): Forward {
  const h = new Array(net.hidden);
  for (let j = 0; j < net.hidden; j++) {
    let z = net.b1[j];
    for (let i = 0; i < net.inDim; i++) z += net.W1[j][i] * x[i];
    h[j] = Math.tanh(z);
  }
  const out = new Array(net.outDim);
  for (let k = 0; k < net.outDim; k++) {
    let z = net.b2[k];
    for (let j = 0; j < net.hidden; j++) z += net.W2[k][j] * h[j];
    out[k] = z;
  }
  return { h, out };
}

export interface Grads {
  gW1: number[][];
  gb1: number[];
  gW2: number[][];
  gb2: number[];
  dIn: number[]; // 对输入的梯度（GAN 把 D 的梯度传回 G 时要用）
}

export function zeroGrads(net: MLP): Grads {
  return {
    gW1: net.W1.map((row) => row.map(() => 0)),
    gb1: net.b1.map(() => 0),
    gW2: net.W2.map((row) => row.map(() => 0)),
    gb2: net.b2.map(() => 0),
    dIn: new Array(net.inDim).fill(0),
  };
}

/** 反向传播：dOut = 损失对输出的梯度；把梯度累加进 g，并算出对输入的梯度 dIn */
export function backwardAccum(net: MLP, x: number[], f: Forward, dOut: number[], g: Grads) {
  const dh = new Array(net.hidden).fill(0);
  for (let k = 0; k < net.outDim; k++) {
    g.gb2[k] += dOut[k];
    for (let j = 0; j < net.hidden; j++) {
      g.gW2[k][j] += dOut[k] * f.h[j];
      dh[j] += dOut[k] * net.W2[k][j];
    }
  }
  for (let j = 0; j < net.hidden; j++) {
    const dz = dh[j] * (1 - f.h[j] * f.h[j]); // tanh'
    g.gb1[j] += dz;
    for (let i = 0; i < net.inDim; i++) {
      g.gW1[j][i] += dz * x[i];
      g.dIn[i] += dz * net.W1[j][i];
    }
  }
}

export function applyGrads(net: MLP, g: Grads, lr: number, scale = 1) {
  for (let j = 0; j < net.hidden; j++) {
    net.b1[j] -= lr * g.gb1[j] * scale;
    for (let i = 0; i < net.inDim; i++) net.W1[j][i] -= lr * g.gW1[j][i] * scale;
  }
  for (let k = 0; k < net.outDim; k++) {
    net.b2[k] -= lr * g.gb2[k] * scale;
    for (let j = 0; j < net.hidden; j++) net.W2[k][j] -= lr * g.gW2[k][j] * scale;
  }
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
