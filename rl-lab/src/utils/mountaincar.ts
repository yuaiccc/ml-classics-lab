export interface MountainCarState {
  position: number;
  velocity: number;
}

const MIN_POS = -1.2;
const MAX_POS = 0.6;
const MAX_SPEED = 0.07;
const GOAL_POS = 0.5;
const GOAL_VEL = 0;
const FORCE = 0.001;
const GRAVITY = 0.0025;

export function step(state: MountainCarState, action: number): MountainCarState {
  const { position, velocity } = state;
  const newVelocity = Math.max(
    -MAX_SPEED,
    Math.min(
      MAX_SPEED,
      velocity + (action - 1) * FORCE - GRAVITY * Math.cos(3 * position)
    )
  );
  const newPosition = Math.max(MIN_POS, Math.min(MAX_POS, position + newVelocity));
  const finalVelocity =
    newPosition === MIN_POS && newVelocity < 0 ? 0 : newVelocity;
  return { position: newPosition, velocity: finalVelocity };
}

export function isTerminal(state: MountainCarState): boolean {
  return state.position >= GOAL_POS;
}

export function randomState(): MountainCarState {
  return {
    position: -0.5 + (Math.random() - 0.5) * 0.1,
    velocity: 0,
  };
}

export function height(pos: number): number {
  return Math.sin(3 * pos) * 0.45 + 0.55;
}

const BAD_WEIGHTS = [
  [0.0, 0.0],
  [0.0, 0.0],
  [0.0, 0.0],
];

const SHAPED_WEIGHTS = [
  [-0.5, 0.3],
  [-0.8, 0.5],
  [-1.2, 0.8],
];

export function linearPolicy(state: MountainCarState, weights: number[]): number {
  const val = weights[0] * state.position + weights[1] * state.velocity;
  if (val < -0.3) return 0;
  if (val > 0.3) return 2;
  return 1;
}

export function randomPolicy(): number {
  return Math.floor(Math.random() * 3);
}

export const WEIGHT_SCHEDULES = {
  dqn: BAD_WEIGHTS,
  shaped: SHAPED_WEIGHTS,
};

export { MIN_POS, MAX_POS, MAX_SPEED, GOAL_POS };
