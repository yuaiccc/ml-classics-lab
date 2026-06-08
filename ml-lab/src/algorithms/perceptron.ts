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
      x: 1 + Math.random() * 2,
      y: 2 + Math.random() * 2,
      label: 0,
    });
  }
  for (let i = 0; i < 40; i++) {
    points.push({
      x: 4 + Math.random() * 2,
      y: 5 + Math.random() * 2,
      label: 1,
    });
  }
  return points;
}

function predict(w1: number, w2: number, b: number, x: number, y: number): number {
  return w1 * x + w2 * y + b >= 0 ? 1 : 0;
}

export function runPerceptron(): Trajectory {
  const points = generateData();
  const lr = 0.5;
  const totalIters = 100;
  const gridX = 50;
  const gridY = 40;
  const xRange: [number, number] = [-1, 8];
  const yRange: [number, number] = [-1, 9];

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
        grid[i][j] = w1 * gx + w2 * gy + b;
      }
    }
    return grid;
  };

  const computeAccuracy = (): number => {
    let correct = 0;
    for (const p of points) {
      if (predict(w1, w2, b, p.x, p.y) === p.label) correct++;
    }
    return correct / points.length;
  };

  const countMistakes = (): number => {
    let mistakes = 0;
    for (const p of points) {
      if (predict(w1, w2, b, p.x, p.y) !== p.label) mistakes++;
    }
    return mistakes;
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
        accuracy: computeAccuracy(),
        mistakes: countMistakes(),
      },
    });

    if (iter < totalIters) {
      let updated = false;
      for (const p of points) {
        const pred = predict(w1, w2, b, p.x, p.y);
        if (pred !== p.label) {
          const sign = p.label === 1 ? 1 : -1;
          w1 += lr * sign * p.x;
          w2 += lr * sign * p.y;
          b += lr * sign;
          updated = true;
        }
      }
      if (!updated) break;
    }
  }

  return {
    meta: {
      id: "perceptron",
      title: "感知机 · Perceptron",
      algorithm: "Perceptron",
      category: "supervised",
      source: "browser",
      abstract:
        "解决线性可分数据的二分类问题：感知机是最简单的神经网络——单层神经元。每当遇到误分类点，就沿该点方向调整权重，决策边界随之「跳变」。与逻辑回归的平滑收敛不同，感知机的边界是阶梯式更新的。",
      description:
        "感知机遇误分类点即更新权重，决策边界阶梯式跳变。数据线性可分时保证收敛。",
      hyperparams: {
        lr: 0.5,
        max_iterations: 100,
        samples: 80,
      },
      insight:
        "感知机收敛定理：若数据线性可分，感知机必在有限步内收敛。但若数据非线性可分，感知机会永远震荡。这个局限直接催生了多层感知机（MLP）——用隐藏层学习非线性特征。感知机是深度学习的起点。",
    },
    frames,
  };
}
