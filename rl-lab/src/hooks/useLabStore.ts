import { create } from "zustand";
import { experiments, Experiment } from "@/data/experiments";

interface LabStore {
  experiments: Experiment[];
  filter: string;
  setFilter: (f: string) => void;
  filteredExperiments: () => Experiment[];
  addExperiment: (exp: Experiment) => void;
}

export const useLabStore = create<LabStore>((set, get) => ({
  experiments,
  filter: "all",
  setFilter: (f) => set({ filter: f }),
  filteredExperiments: () => {
    const { experiments, filter } = get();
    if (filter === "all") return experiments;
    return experiments.filter((e) => e.algorithm === filter);
  },
  addExperiment: (exp) =>
    set((state) => ({ experiments: [...state.experiments, exp] })),
}));
