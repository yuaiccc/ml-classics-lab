// 通用时间轴播放：接收一个 Trajectory，提供播放/暂停/单步/拖拽/调速。
// 所有算法家族共用，前端只管回放（见 ML_LAB_DESIGN.md §1）。

import { useCallback, useEffect, useState } from "react";
import { Frame, Trajectory } from "./types";

export function useTrajectory<S>(traj: Trajectory<S>, baseFps = 6) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const last = traj.frames.length - 1;

  // 轨迹本身变化（重新生成数据）时复位
  useEffect(() => {
    setIndex(0);
    setPlaying(false);
  }, [traj]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setIndex((i) => {
        if (i >= last) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 1000 / (baseFps * speed));
    return () => clearInterval(id);
  }, [playing, speed, last, baseFps]);

  const play = useCallback(() => {
    setIndex((i) => (i >= last ? 0 : i)); // 播完后再按播放则从头开始
    setPlaying(true);
  }, [last]);
  const pause = useCallback(() => setPlaying(false), []);
  const reset = useCallback(() => {
    setPlaying(false);
    setIndex(0);
  }, []);
  const stepFwd = useCallback(() => {
    setPlaying(false);
    setIndex((i) => Math.min(last, i + 1));
  }, [last]);
  const stepBack = useCallback(() => {
    setPlaying(false);
    setIndex((i) => Math.max(0, i - 1));
  }, []);
  const seek = useCallback(
    (i: number) => {
      setPlaying(false);
      setIndex(Math.max(0, Math.min(last, i)));
    },
    [last]
  );

  // 切换到帧数更少的轨迹时，复位 effect 尚未执行，需在渲染期先钳住 index，
  // 否则 traj.frames[index] 会越界为 undefined 导致崩溃。
  const safeIndex = Math.min(index, last);
  const frame: Frame<S> = traj.frames[safeIndex];
  return {
    index: safeIndex,
    frame,
    last,
    playing,
    speed,
    setSpeed,
    play,
    pause,
    reset,
    stepFwd,
    stepBack,
    seek,
  };
}
