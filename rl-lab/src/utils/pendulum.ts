export interface PendulumState {
  theta: number;
  thetaDot: number;
}

const G = 10.0;
const M = 1.0;
const L = 1.0;
const DT = 0.05;
const MAX_SPEED = 8.0;
const MAX_TORQUE = 2.0;

export function step(state: PendulumState, torque: number): PendulumState {
  const clampedTorque = Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, torque));
  const newThetaDot =
    Math.max(-MAX_SPEED, Math.min(MAX_SPEED, state.thetaDot + (3 * G / (2 * L) * Math.sin(state.theta) + 3 / (M * L * L) * clampedTorque) * DT));
  const newTheta = state.theta + newThetaDot * DT;
  return { theta: normalize(newTheta), thetaDot: newThetaDot };
}

function normalize(angle: number): number {
  while (angle > Math.PI) angle -= 2 * Math.PI;
  while (angle < -Math.PI) angle += 2 * Math.PI;
  return angle;
}

export function randomState(): PendulumState {
  return {
    theta: (Math.random() - 0.5) * 2 * Math.PI,
    thetaDot: (Math.random() - 0.5) * 4,
  };
}

export function reward(state: PendulumState, action: number): number {
  const torque = Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, action));
  return -(state.theta ** 2 + 0.1 * state.thetaDot ** 2 + 0.001 * torque ** 2);
}

const BAD_WEIGHTS = [0.0, 0.0];
const MID_WEIGHTS = [3.0, 0.8];
const GOOD_WEIGHTS = [8.0, 2.0];

export function sacPolicy(state: PendulumState, weights: number[], noise: number = 0): number {
  const torque = -(weights[0] * state.theta + weights[1] * state.thetaDot);
  return Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, torque + (Math.random() - 0.5) * noise));
}

export const WEIGHT_SCHEDULES = [BAD_WEIGHTS, MID_WEIGHTS, GOOD_WEIGHTS];

export { MAX_TORQUE };
