import { Trajectory, Frame, ScatterBoundaryState } from "@/player/types";

interface Point2D {
  x: number;
  y: number;
  label: number;
}

function generateData(): Point2D[] {
  const points: Point2D[] = [];
  for (let i = 0; i < 30; i++) {
    points.push({
      x: 1.5 + Math.random() * 1.5,
      y: 3.5 + Math.random() * 1.5,
      label: 0,
    });
  }
  for (let i = 0; i < 30; i++) {
    points.push({
      x: 4 + Math.random() * 1.5,
      y: 6 + Math.random() * 1.5,
      label: 1,
    });
  }
  for (let i = 0; i < 20; i++) {
    const x = 2.5 + Math.random() * 2;
    const y = 4.5 + Math.random() * 2;
    const label = x + y > 7.5 ? 1 : 0;
    points.push({ x, y, label });
  }
  return points;
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
}

export function runLogisticRegression(): Trajectory {
  const points = generateData();
  const lr = 0.05;
  const totalIters = 300;
  const gridX = 50;
  const gridY = 40;
  const xRange: [number, number] = [0, 7];
  const yRange: [number, number] = [2, 9];

  let w1 = (Math.random() - 0.5) * 0.5;
  let w2 = (Math.random() - 0.5) * 0.5;
  let b = 0;

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
      id: "logistic-regression",
      title: "逻辑回归 · Logistic Regression",
      algorithm: "Logistic Regression",
      category: "supervised",
      source: "browser",
      abstract:
        "解决二分类问题：逻辑回归用 sigmoid 函数将线性组合映射到 [0,1] 概率，通过交叉熵损失驱动梯度下降，让决策边界从随机位置逐步成形到正确分隔两类数据的位置。与梯度下降实验不同，这里的数据有重叠区域，决策边界需要处理噪声。",
      description:
        "逻辑回归处理有噪声重叠的二分类数据，300 次迭代后决策边界精确成形。",
      hyperparams: {
        lr: 0.05,
        iterations: 300,
        samples: 80,
        optimizer: "SGD",
      },
      insight:
        "逻辑回归的本质：在线性组合 w·x+b 上套 sigmoid 得到概率 P(y=1|x)，用交叉熵损失衡量预测与真实的差距。决策边界是 w·x+b=0 的超平面（2D 中是一条直线）。对于非线性可分数据，需要特征变换或神经网络。",
    },
    frames,
  };
}
