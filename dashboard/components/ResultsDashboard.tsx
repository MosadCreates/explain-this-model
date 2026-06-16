"use client";

import { useEffect } from "react";
import { useAnalysisStore } from "@/store/analysis";
import { useResult } from "@/lib/api";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import NeuronView from "./NeuronView";
import AttentionView from "./AttentionView";
import JobStatusCard from "./JobStatusCard";

export default function ResultsDashboard() {
  const { jobId, status, setStatus, setResult, setError } = useAnalysisStore();
  const { data, error, isLoading } = useResult(status === "running" ? jobId : null);

  useEffect(() => {
    if (data?.status === "completed" && data.result) {
      setStatus("completed");
      setResult(data.result);
    } else if (data?.status === "failed") {
      setStatus("failed");
      setError(data.error_message || "Analysis failed");
    }
  }, [data, setStatus, setResult, setError]);

  if (status === "running") {
    return jobId ? <JobStatusCard jobId={jobId} /> : null;
  }

  if (status === "failed") {
    const msg = useAnalysisStore.getState().error;
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle className="text-lg text-destructive">Analysis Failed</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-destructive">{msg || "Unknown error"}</p>
        </CardContent>
      </Card>
    );
  }

  if (status === "completed") {
    const storeResult = useAnalysisStore.getState().result;
    const result = data?.result || storeResult;
    if (!result) {
      return (
        <Card>
          <CardContent className="pt-6">
            <Skeleton className="h-20 w-full" />
          </CardContent>
        </Card>
      );
    }

    const neurons = result.neuron_results || [];
    const heads = result.attention_results || [];
    const explanations = result.explanations;
    const neuronExps = explanations?.neurons || [];
    const headExps = explanations?.heads || [];

    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Results</span>
              <span className="text-sm font-normal text-muted-foreground">
                {result.architecture_type} &middot; {result.model_name}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold">{result.neuron_count}</div>
                <div className="text-xs text-muted-foreground">Neurons</div>
              </div>
              <div>
                <div className="text-2xl font-bold">{result.head_count}</div>
                <div className="text-xs text-muted-foreground">Attention Heads</div>
              </div>
              <div>
                <div className="text-2xl font-bold">{result.total_dead_neurons}</div>
                <div className="text-xs text-muted-foreground">Dead Neurons</div>
              </div>
              <div>
                <div className="text-2xl font-bold">{result.analysis_duration_seconds.toFixed(1)}s</div>
                <div className="text-xs text-muted-foreground">Analysis Time</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="flex items-center gap-2 text-xs text-muted-foreground bg-muted rounded-lg px-3 py-2">
          <span>API calls: {explanations?.total_api_calls ?? 0}</span>
          <span>&middot;</span>
          <span>Cached: {explanations?.total_cached ?? 0}</span>
          <span>&middot;</span>
          <span>Explanations: {explanations?.explanation_duration_seconds?.toFixed(1) ?? 0}s</span>
        </div>

        <Tabs defaultValue="neurons">
          <TabsList className="w-full">
            <TabsTrigger value="neurons" className="flex-1">
              Neurons ({neurons.length})
            </TabsTrigger>
            <TabsTrigger value="attention" className="flex-1">
              Attention ({heads.length})
            </TabsTrigger>
          </TabsList>
          <TabsContent value="neurons">
            <NeuronView neurons={neurons} explanations={neuronExps} />
          </TabsContent>
          <TabsContent value="attention">
            <AttentionView heads={heads} explanations={headExps} />
          </TabsContent>
        </Tabs>
      </div>
    );
  }

  return null;
}
