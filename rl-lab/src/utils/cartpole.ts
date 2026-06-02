export interface CartPoleState {
  x: number;
  xDot: number;
  theta: number;
  thetaDot: number;
}

const GRAVITY = 9.8;
const MASS_CART = 1.0;
const MASS_POLE = 0.1;
const TOTAL_MASS = MASS_CART + MASS_POLE;
const HALF_POLE_LEN = 0.5;
const POLEMASS_LENGTH = MASS_POLE * HALF_POLE_LEN;
const FORCE_MAG = 10.0;
const DT = 0.02;
const X_THRESHOLD = 4.8;
const THETA_THRESHOLD = 24 * (Math.PI / 180);

const OPTIMAL_WEIGHTS = [0.2, 0.8, 2.5, 1.8];

export function step(
  state: CartPoleState,
  action: number
): CartPoleState {
  const force = action === 1 ? FORCE_MAG : -FORCE_MAG;
  const { x, xDot, theta, thetaDot } = state;

  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);

  const temp =
    (force + POLEMASS_LENGTH * thetaDot * thetaDot * sinTheta) / TOTAL_MASS;
  const thetaAcc =
    (GRAVITY * sinTheta - cosTheta * temp) /
    (HALF_POLE_LEN * (4.0 / 3.0 - (MASS_POLE * cosTheta * cosTheta) / TOTAL_MASS));
  const xAcc = temp - POLEMASS_LENGTH * thetaAcc * cosTheta / TOTAL_MASS;

  return {
    x: x + xDot * DT + 0.5 * xAcc * DT * DT,
    xDot: xDot + xAcc * DT,
    theta: theta + thetaDot * DT + 0.5 * thetaAcc * DT * DT,
    thetaDot: thetaDot + thetaAcc * DT,
  };
}

export function isTerminal(state: CartPoleState): boolean {
  return (
    state.x < -X_THRESHOLD ||
    state.x > X_THRESHOLD ||
    state.theta < -THETA_THRESHOLD ||
    state.theta > THETA_THRESHOLD
  );
}

export { X_THRESHOLD, THETA_THRESHOLD, OPTIMAL_WEIGHTS };
