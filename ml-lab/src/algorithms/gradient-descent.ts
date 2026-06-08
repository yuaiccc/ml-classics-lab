import { Trajectory, Frame, ScatterBoundaryState } from "@/player/types";

interface Point2D {
  x: number;
  y: number;
  label: number;
}

function generateData(): Point2D[] {
  const points: Point2D[] = [];
  for (let i = 0; i < 40; i++) {
    points.push({
      x: 1 + Math.random() * 2 - 1,
      y: 2 + Math.random() * 2 - 1,
      label: 0,
    });
  }
  for (let i = 0; i < 40; i++) {
    points.push({
      x: 4 + Math.random() * 2 - 1,
      y: 5 + Math.random() * 2 - 1,
      label: 1,
    });
  }
  return points;
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
}

export function runGradientDescent(): Trajectory {
  const points = generateData();
  const lr = 0.1;
  const totalIters = 200;
  const gridX = 50;
  const gridY = 40;
  const xRange: [number, number] = [-1, 7];
  const yRange: [number, number] = [-1, 8];

  let w1 = (Math.random() - 0.5) * 2;
  let w2 = (Math.random() - 0.5) * 2;
  let b = (Math.random() - 0.5) * 2;

  const frames: Frame[] = [];

  const computeBoundary = (): number[][] => {
    const grid: number[][] = [];
    for (let i = 0; i < gridX; i++) {
      grid[i] = [];
      const gx = xRange[0] + (i / gridX) * (xRange[1] - xRange[0]);
      for (let j = 0; j < gridY; j++) {
        const gy = yRange[0] + (j / gridY) * (yRange[1] - yRange[0]);
        grid[i][j] = sigmoid(w1 * gx + w2 * gy + b);
      }
    }
    return grid;
  };

  const computeLoss = (): number => {
    let loss = 0;
    for (const p of points) {
      const pred = sigmoid(w1 * p.x + w2 * p.y + b);
      loss += -p.label * Math.log(pred + 1e-8) - (1 - p.label) * Math.log(1 - pred + 1e-8);
    }
    return loss / points.length;
  };

  const computeAccuracy = (): number => {
    let correct = 0;
    for (const p of points) {
      const pred = sigmoid(w1 * p.x + w2 * p.y + b) >= 0.5 ? 1 : 0;
      if (pred === p.label) correct++;
    }
    return correct / points.length;
  };

  for (let iter = 0; iter <= totalIters; iter++) {
    const stateData: ScatterBoundaryState = {
      points,
      boundary: computeBoundary(),
      gridX,
      gridY,
      xRange,
      yRange,
    };

    frames.push({
      iter,
      state: { family: "scatter-boundary", data: stateData },
      metrics: {
        loss: computeLoss(),
        accuracy: computeAccuracy(),
      },
    });

    if (iter < totalIters) {
      let dw1 = 0, dw2 = 0, db = 0;
      for (const p of points) {
        const pred = sigmoid(w1 * p.x + w2 * p.y + b);
        const err = pred - p.label;
        dw1 += err * p.x;
        dw2 += err * p.y;
        db += err;
      }
      w1 -= lr * dw1 / points.length;
      w2 -= lr * dw2 / points.length;
      b -= lr * db / points.length;
    }
  }

  return {
    meta: {
      id: "gradient-descent",
      title: "梯度下降 · 逻辑回归",
      algorithm: "Gradient Descent",
      category: "supervised",
      source: "browser",
      abstract:
        "解决二分类问题中的决策边界学习：如何通过梯度下降逐步调整参数，使逻辑回归的决策边界从随机位置收敛到正确分类两类数据点的位置。这是所有神经网络训练的基础——损失函数上的梯度指引参数更新方向。",
      description:
        "逻辑回归通过梯度下降学习决策边界。初始边界随机，200 次迭代后精确分隔两类点。",
      hyperparams: {
        lr: 0.1,
        iterations: 200,
        samples: 80,
        optimizer: "SGD",
      },
      insight:
        "梯度下降的核心：计算损失函数对每个参数的偏导数（梯度），沿梯度反方向更新参数。学习率太大→震荡不收敛；太小→收敛太慢。逻辑回归的决策边界是线性的（一条直线），更复杂的数据需要神经网络。",
    },
    frames,
  };
}
