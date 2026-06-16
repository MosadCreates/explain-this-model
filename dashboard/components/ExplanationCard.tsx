"use client";

import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ExplanationCardProps {
  label: string;
  hypothesis: string;
  confidence: string;
  pattern_type: string;
  cached: boolean;
  isNeuron?: boolean;
}

function confidenceVariant(confidence: string) {
  switch (confidence.toLowerCase()) {
    case "high": return "default" as const;
    case "medium": return "secondary" as const;
    default: return "outline" as const;
  }
}

function patternBadgeVariant(pattern: string) {
  switch (pattern) {
    case "diagonal":
    case "previous_token":
    case "first_token": return "secondary" as const;
    case "content_based":
    case "induction_head": return "default" as const;
    case "diffuse": return "outline" as const;
    default: return "outline" as const;
  }
}

export default function ExplanationCard({
  label,
  hypothesis,
  confidence,
  pattern_type,
  cached,
}: ExplanationCardProps) {
  return (
    <TooltipProvider>
      <div className="rounded-lg border p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono text-muted-foreground">{label}</span>
          <div className="flex items-center gap-1.5">
            <Badge variant={confidenceVariant(confidence)} className="text-[10px] px-1.5 py-0">
              {confidence}
            </Badge>
            <Badge variant={patternBadgeVariant(pattern_type)} className="text-[10px] px-1.5 py-0">
              {pattern_type}
            </Badge>
            {cached && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-[10px] text-muted-foreground cursor-help">(cached)</span>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-xs">Explanation reused from cache</p>
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>
        <p className="text-sm leading-relaxed">{hypothesis}</p>
      </div>
    </TooltipProvider>
  );
}
