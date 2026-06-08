import { ExperimentLegacy } from "@/data/experiments";

export interface ClassicProblem {
  id: string;
  title: string;
  category: ExperimentLegacy["category"];
  domain: string;
  benchmark: string;
  metric: string;
  whyClassic: string;
  commonMethods: string[];
  plannedRuns: string[];
  history: {
    problemYear: number;
    problemLabel: string;
    breakthroughYear: number;
    breakthroughLabel: string;
  };
}

export const classicProblems: ClassicProblem[] = [
];
