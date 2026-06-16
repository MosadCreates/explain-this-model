"use client";

import { Textarea } from "@/components/ui/textarea";

interface PromptInputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

const MAX_CHARS = 4000;

export default function PromptInput({ value, onChange, disabled }: PromptInputProps) {
  return (
    <div className="space-y-2">
      <Textarea
        placeholder="Enter your prompt here... e.g. 'The transformer model processes language by'"
        value={value}
        onChange={(e) => onChange(e.target.value.slice(0, MAX_CHARS))}
        disabled={disabled}
        rows={4}
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>Write a prompt to analyse neuron activations</span>
        <span className={value.length > MAX_CHARS * 0.9 ? "text-destructive" : ""}>
          {value.length}/{MAX_CHARS}
        </span>
      </div>
    </div>
  );
}
