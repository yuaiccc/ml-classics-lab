import { Trajectory, Frame, ScatterBoundaryState } from "@/player/types";

interface Point2D {
  x: number;
  y: number;
  label: number;
}

function generateData(): Point2D[] {
  const points: Point2D[] = [];
  const slope = 1.8;
  const intercept = 1.0;
  for (let i = 0; i < 60; i++) {
    const x = Math.random() * 8;
    const noise = (Math.random() - 0.5) * 2.5;
    const y = slope * x + intercept + noise;
    points.push({ x, y, label: 0 });
  }
  return points;
}

export function runLinearRegression(): Trajectory {
  const points = generateData();
  const lr = 0.005;
  const totalIters = 300;
  const gridX = 50;
  const gridY = 40;
  const xRange: [number, number] = [-1, 9];
  const yRange: [number, number] = [-3, 20];

  let w = (Math.random() - 0.5) * 2;
  let b = (Math.random() - 0.5) * 5;

  const frames: Frame[] = [];

  const computeLoss = (): number => {
    let loss = 0;
    for (const p of points) {
      const pred = w * p.x + b;
      loss += (pred - p.y) ** 2;
    }
    return loss / (2 * points.length);
  };

  const computeR2 = (): number => {
    const meanY = points.reduce((s, p) => s + p.y, 0) / points.length;
    let ssTot = 0;
    let ssRes = 0;
    for (const p of points) {
      ssTot += (p.y - meanY) ** 2;
      ssRes += (p.y - (w * p.x + b)) ** 2;
    }
    return 1 - ssRes / ssTot;
  };

  for (let iter = 0; iter <= totalIters; iter++) {
    const stateData: ScatterBoundaryState = {
      points,
      boundary: [],
      gridX,
      gridY,
      xRange,
      yRange,
      fit: { slope: w, intercept: b },
    };

    frames.push({
      iter,
      state: { family: "scatter-boundary", data: stateData },
      metrics: {
        mse: computeLoss(),
        r2: computeR2(),
      },
    });

    if (iter < totalIters) {
      let dw = 0;
      let db = 0;
      for (const p of points) {
        const pred = w * p.x + b;
        const err = pred - p.y;
        dw += err * p.x;
        db += err;
      }
      w -= lr * dw / points.length;
      b -= lr * db / points.length;
    }
  }

  return {
    meta: {
      id: "linear-regression",
      title: "线性回归 · Linear Regression",
      algorithm: "Linear Regression",
      category: "supervised",
      source: "browser",
      abstract:
        "解决连续值预测问题：给定一组带噪声的 (x, y) 数据点，如何找到一条直线 y=wx+b 最佳拟合数据？线性回归通过最小化均方误差（MSE），让拟合线从随机位置逐帧旋转贴合到数据趋势上。这是回归问题的基石。",
      description:
        "线性回归用梯度下降最小化 MSE，拟合线逐帧旋转贴合数据。300 次迭代后收敛。",
      hyperparams: {
        lr: 0.005,
        iterations: 300,
        samples: 60,
        optimizer: "SGD",
      },
      insight:
        "线性回归的损失函数 MSE 是凸函数——只有一个全局最小值，梯度下降保证收敛。R² 越接近 1 表示拟合越好。线性回归假设 y 与 x 是线性关系，对于非线性关系需要多项式特征或非线性模型。",
    },
    frames,
  };
}
