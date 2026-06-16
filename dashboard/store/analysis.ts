import { create } from "zustand";
import type { AnalysisResult } from "@/lib/types";

export type AppStatus = "idle" | "submitting" | "running" | "completed" | "failed";

interface AnalysisState {
  modelName: string;
  prompt: string;
  jobId: string | null;
  status: AppStatus;
  error: string | null;
  result: AnalysisResult | null;

  setModelName: (name: string) => void;
  setPrompt: (prompt: string) => void;
  setJobId: (id: string | null) => void;
  setStatus: (status: AppStatus) => void;
  setError: (error: string | null) => void;
  setResult: (result: AnalysisResult | null) => void;
  reset: () => void;
}

const initialState = {
  modelName: "",
  prompt: "",
  jobId: null,
  status: "idle" as AppStatus,
  error: null,
  result: null,
};

export const useAnalysisStore = create<AnalysisState>((set) => ({
  ...initialState,
  setModelName: (modelName) => set({ modelName }),
  setPrompt: (prompt) => set({ prompt }),
  setJobId: (jobId) => set({ jobId }),
  setStatus: (status) => set({ status }),
  setError: (error) => set({ error }),
  setResult: (result) => set({ result }),
  reset: () => set(initialState),
}));
