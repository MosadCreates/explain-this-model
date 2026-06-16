import useSWR, { mutate } from "swr";
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  JobStatusResponse,
  ModelSearchResult,
  ResultResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function fetcher<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(body || res.statusText, res.status);
  }
  return res.json();
}

async function poster<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(text || res.statusText, res.status);
  }
  return res.json();
}

export function submitAnalysis(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  return poster<AnalyzeResponse>(`${API_BASE}/analyze`, req);
}

export function useJobStatus(jobId: string | null) {
  return useSWR<JobStatusResponse>(
    jobId ? `${API_BASE}/jobs/${jobId}` : null,
    fetcher,
    {
      refreshInterval: (data) =>
        data?.status === "completed" || data?.status === "failed" ? 0 : 1000,
      revalidateOnFocus: false,
    }
  );
}

export function useResult(jobId: string | null) {
  return useSWR<ResultResponse>(
    jobId ? `${API_BASE}/results/${jobId}` : null,
    fetcher,
    {
      refreshInterval: (data) =>
        data?.status === "completed" || data?.status === "failed" ? 0 : 2000,
      revalidateOnFocus: false,
    }
  );
}

export function searchModels(query: string): Promise<ModelSearchResult[]> {
  return fetcher<{ results: ModelSearchResult[] }>(
    `${API_BASE}/models/search?query=${encodeURIComponent(query)}&limit=8`
  ).then((r) => r.results);
}

export function useModelSearch(query: string) {
  return useSWR<ModelSearchResult[]>(
    query.length >= 2 ? `model-search:${query}` : null,
    () => searchModels(query),
    { revalidateOnFocus: false }
  );
}

export function useHealth() {
  return useSWR<{ status: string; version: string }>(
    `${API_BASE}/health`,
    fetcher,
    { revalidateOnFocus: false }
  );
}

export function invalidateJob(jobId: string) {
  mutate(`${API_BASE}/jobs/${jobId}`);
  mutate(`${API_BASE}/results/${jobId}`);
}

export async function validateModel(modelName: string): Promise<{ valid: boolean; parameter_count?: number; architecture?: string }> {
  return fetcher(`${API_BASE}/models/validate?model_name=${encodeURIComponent(modelName)}`);
}
