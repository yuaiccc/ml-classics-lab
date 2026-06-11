// 激活函数与其导数（纯函数，供 builder 与 Viz 复用、单测覆盖）。
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

export const relu = (x: number): number => (x > 0 ? x : 0);
export const dRelu = (x: number): number => (x > 0 ? 1 : 0);

// SiLU(x) = x * sigmoid(x)
export const silu = (x: number): number => x * sigmoid(x);
export const dSilu = (x: number): number => {
  const s = sigmoid(x);
  return s + x * s * (1 - s);
};

// Swish(x) = x * sigmoid(beta*x)（beta=1 即 SiLU）
export const swish = (x: number, beta = 1): number => x * sigmoid(beta * x);
export const dSwish = (x: number, beta = 1): number => {
  const s = sigmoid(beta * x);
  return s + beta * x * s * (1 - s);
};

// GELU（tanh 近似，与原论文一致）
export const gelu = (x: number): number =>
  0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
export const dGelu = (x: number): number => {
  const h = 1e-5;
  return (gelu(x + h) - gelu(x - h)) / (2 * h);
};
