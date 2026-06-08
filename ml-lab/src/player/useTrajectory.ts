import { useState, useCallback, useRef, useEffect } from "react";
import { Trajectory } from "./types";

export interface TrajectoryPlayerState {
  currentFrame: number;
  totalFrames: number;
  playing: boolean;
  speed: number;
}

export function useTrajectory(trajectory: Trajectory) {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const animRef = useRef<number>(0);
  const lastTimeRef = useRef(0);

  const totalFrames = trajectory.frames.length;
  const frame = trajectory.frames[currentFrame] ?? trajectory.frames[0];

  const goTo = useCallback((idx: number) => {
    setCurrentFrame(Math.max(0, Math.min(idx, totalFrames - 1)));
  }, [totalFrames]);

  const stepForward = useCallback(() => {
    setCurrentFrame((i) => Math.min(i + 1, totalFrames - 1));
  }, [totalFrames]);

  const stepBackward = useCallback(() => {
    setCurrentFrame((i) => Math.max(i - 1, 0));
  }, []);

  const reset = useCallback(() => {
    setCurrentFrame(0);
    setPlaying(false);
  }, []);

  useEffect(() => {
    if (!playing) {
      cancelAnimationFrame(animRef.current);
      return;
    }

    const FRAME_MS = 1000 / (30 * speed);

    const loop = (time: number) => {
      if (time - lastTimeRef.current < FRAME_MS) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }
      lastTimeRef.current = time;

      setCurrentFrame((i) => {
        if (i >= totalFrames - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing, speed, totalFrames]);

  return {
    frame,
    currentFrame,
    totalFrames,
    playing,
    speed,
    setPlaying,
    setSpeed,
    goTo,
    stepForward,
    stepBackward,
    reset,
  };
}
