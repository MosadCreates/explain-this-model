"use client";

import { useEffect } from "react";
import { useParams } from "next/navigation";
import Layout from "@/components/Layout";
import JobStatusCard from "@/components/JobStatusCard";
import ResultsDashboard from "@/components/ResultsDashboard";
import { useJobStatus } from "@/lib/api";
import { useAnalysisStore } from "@/store/analysis";

export default function AnalyzePage() {
  const params = useParams();
  const jobId = params.jobId as string;
  const { status, setJobId, setStatus } = useAnalysisStore();
  const { data: statusData } = useJobStatus(jobId);

  useEffect(() => {
    setJobId(jobId);
    setStatus("running");
    return () => {
      setJobId(null);
      setStatus("idle");
    };
  }, [jobId, setJobId, setStatus]);

  useEffect(() => {
    if (statusData?.status === "completed" || statusData?.status === "failed") {
      setStatus(statusData.status);
    }
  }, [statusData?.status, setStatus]);

  return (
    <Layout>
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="text-center mb-2">
          <h1 className="text-2xl font-bold tracking-tight">Analysis Job</h1>
          <p className="text-sm text-muted-foreground font-mono">{jobId}</p>
        </div>
        {(status === "running" || status === "pending") && <JobStatusCard jobId={jobId} />}
        {status === "completed" && <ResultsDashboard />}
        {status === "failed" && (
          <JobStatusCard jobId={jobId} />
        )}
      </div>
    </Layout>
  );
}
