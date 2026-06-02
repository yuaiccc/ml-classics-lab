// 极简线性代数：高斯消元解线性方程 + 岭回归多项式拟合。供过拟合/正则化实验用。

export function gaussianSolve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / d;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-12));
}

/** 岭回归多项式拟合：返回系数 [c0, c1, ..., c_deg]，最小化 Σ(y-φ·w)² + λ‖w‖² */
export function polyfit(xs: number[], ys: number[], deg: number, lambda = 0): number[] {
  const n = xs.length;
  const m = deg + 1;
  const phi = xs.map((x) => {
    const r: number[] = [];
    let p = 1;
    for (let j = 0; j < m; j++) {
      r.push(p);
      p *= x;
    }
    return r;
  });
  const XtX = Array.from({ length: m }, () => new Array(m).fill(0));
  const Xty = new Array(m).fill(0);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < m; a++) {
      Xty[a] += phi[i][a] * ys[i];
      for (let b = 0; b < m; b++) XtX[a][b] += phi[i][a] * phi[i][b];
    }
  }
  for (let a = 0; a < m; a++) XtX[a][a] += lambda;
  return gaussianSolve(XtX, Xty);
}

export const polyval = (w: number[], x: number): number => {
  let p = 1;
  let s = 0;
  for (const c of w) {
    s += c * p;
    p *= x;
  }
  return s;
};
