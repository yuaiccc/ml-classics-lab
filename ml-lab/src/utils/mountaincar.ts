export interface MountainCarState {
  position: number;
  velocity: number;
}

const MIN_POS = -1.2;
const MAX_POS = 0.6;
const MAX_SPEED = 0.07;
const GOAL_POS = 0.5;
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

export function randomPolicy(): number {
  return Math.floor(Math.random() * 3);
}

export function momentumPolicy(state: MountainCarState, noise: number = 0): number {
  if (Math.random() < noise) return Math.floor(Math.random() * 3);
  if (state.velocity < 0) return 0;
  return 2;
}

export type EpochSchedule = {
  label: string;
  sub: string;
  color: string;
  policy: (state: MountainCarState) => number;
};

export const DQN_EPOCH_SCHEDULES: EpochSchedule[] = [
  {
    label: "随机探索",
    sub: "完全随机",
    color: "#ff5252",
    policy: () => randomPolicy(),
  },
  {
    label: "随机探索",
    sub: "仍无信号",
    color: "#ff5252",
    policy: () => randomPolicy(),
  },
  {
    label: "随机探索",
    sub: "持续失败",
    color: "#ff5252",
    policy: () => randomPolicy(),
  },
];

export const SHAPED_EPOCH_SCHEDULES: EpochSchedule[] = [
  {
    label: "初期",
    sub: "随机+少量引导",
    color: "#ff5252",
    policy: (s) => momentumPolicy(s, 0.7),
  },
  {
    label: "中期",
    sub: "动量策略",
    color: "#ffab40",
    policy: (s) => momentumPolicy(s, 0.2),
  },
  {
    label: "后期",
    sub: "精确控制",
    color: "#00ff88",
    policy: (s) => momentumPolicy(s, 0),
  },
];

export { MIN_POS, MAX_POS, MAX_SPEED, GOAL_POS };
