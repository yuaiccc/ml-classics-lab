import { create } from "zustand";
import { experiments, ExperimentLegacy } from "@/data/experiments";

interface LabStore {
  experiments: ExperimentLegacy[];
  filter: string;
  setFilter: (f: string) => void;
  filteredExperiments: () => ExperimentLegacy[];
}

export const useLabStore = create<LabStore>((set, get) => ({
  experiments,
  filter: "all",
  setFilter: (f) => set({ filter: f }),
  filteredExperiments: () => {
    const { experiments, filter } = get();
    if (filter === "all") return experiments;
    return experiments.filter((e) => e.category === filter);
  },
}));
