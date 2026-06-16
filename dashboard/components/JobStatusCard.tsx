"use client";

import { useJobStatus } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface JobStatusCardProps {
  jobId: string;
}

const stages = [
  { key: "pending", label: "Loading Model", value: 15 },
  { key: "running", label: "Extracting Activations", value: 45 },
  { key: "analyzing", label: "Analyzing Results", value: 65 },
  { key: "explaining", label: "Generating Explanations", value: 85 },
  { key: "completed", label: "Complete", value: 100 },
];

function getStage(status: string): number {
  switch (status) {
    case "pending": return 0;
    case "running": return 1;
    case "analyzing": return 2;
    case "explaining": return 3;
    case "completed": return 4;
    default: return 0;
  }
}

export default function JobStatusCard({ jobId }: JobStatusCardProps) {
  const { data, error, isLoading } = useJobStatus(jobId);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Job Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            <div className="h-4 bg-muted rounded w-3/4" />
            <div className="h-4 bg-muted rounded w-1/2" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle className="text-lg text-destructive">Error</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-destructive">{error.message}</p>
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  const currentStage = getStage(data.status);
  const progressValue = data.status === "completed" ? 100 : stages[currentStage]?.value ?? 45;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <span className={statusColor(data.status)}>{data.status}</span>
          {data.status === "running" && <LoadingDots />}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Model</span>
          <span className="font-mono text-xs">{data.model_name}</span>
        </div>
        <div className="space-y-2">
          {data.status !== "completed" && data.status !== "failed" && (
            <>
              <Progress value={progressValue} className="mt-2" />
              <div className="flex justify-between text-[10px] text-muted-foreground">
                {stages.slice(0, -1).map((stage, i) => (
                  <span
                    key={stage.key}
                    className={cn(
                      i <= currentStage ? "text-primary font-medium" : "text-muted-foreground"
                    )}
                  >
                    {stage.label}
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
        {data.status === "failed" && data.error_message && (
          <p className="text-xs text-destructive mt-2">{data.error_message}</p>
        )}
        {data.status === "completed" && (
          <p className="text-xs text-green-600 mt-2">Analysis complete. View results below.</p>
        )}
      </CardContent>
    </Card>
  );
}

function LoadingDots() {
  return (
    <span className="inline-flex gap-0.5">
      <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
      <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
      <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
    </span>
  );
}

function statusColor(status: string): string {
  switch (status) {
    case "pending": return "text-yellow-600";
    case "running": return "text-blue-600";
    case "completed": return "text-green-600";
    case "failed": return "text-destructive";
    default: return "";
  }
}
