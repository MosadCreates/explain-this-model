"use client";

import { useState, useRef, useEffect } from "react";
import { useModelSearch, validateModel } from "@/lib/api";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface ModelSearchProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export default function ModelSearch({ value, onChange, disabled }: ModelSearchProps) {
  const [query, setQuery] = useState(value);
  const [debounced, setDebounced] = useState("");
  const [open, setOpen] = useState(false);
  const [validating, setValidating] = useState(false);
  const [validationMsg, setValidationMsg] = useState<string | null>(null);
  const [valid, setValid] = useState<boolean | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const blurTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setQuery(value);
  }, [value]);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(query), 300);
    return () => clearTimeout(timer);
  }, [query]);

  const { data: results } = useModelSearch(debounced);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  async function handleBlur() {
    const name = query.trim();
    if (!name || name.length < 2) return;
    blurTimer.current = setTimeout(async () => {
      setValidating(true);
      try {
        const res = await validateModel(name);
        setValid(res.valid);
        if (res.valid) {
          let info = `Valid \u2713`;
          if (res.parameter_count) {
            info += ` (${(res.parameter_count / 1e6).toFixed(0)}M params)`;
          }
          setValidationMsg(info);
        } else {
          setValidationMsg("Model not found on HuggingFace");
        }
      } catch {
        setValid(false);
        setValidationMsg("Could not validate model");
      } finally {
        setValidating(false);
      }
    }, 500);
  }

  function handleFocus() {
    if (blurTimer.current) clearTimeout(blurTimer.current);
    setValidationMsg(null);
    setValid(null);
    if (query.length >= 2) setOpen(true);
  }

  return (
    <div ref={ref} className="relative">
      <div className="relative">
        <Input
          placeholder="e.g. gpt2, distilbert-base-uncased, facebook/opt-350m"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            onChange(e.target.value);
            if (e.target.value.length >= 2) setOpen(true);
            setValidationMsg(null);
            setValid(null);
          }}
          onFocus={handleFocus}
          onBlur={handleBlur}
          disabled={disabled}
          className={cn(
            valid === true && "border-green-500",
            valid === false && "border-red-500"
          )}
        />
        {validating && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground animate-pulse">
            Validating...
          </span>
        )}
        {!validating && validationMsg && (
          <span className={cn(
            "absolute right-3 top-1/2 -translate-y-1/2 text-xs",
            valid ? "text-green-600" : "text-red-500"
          )}>
            {validationMsg}
          </span>
        )}
      </div>
      {open && results && results.length > 0 && (
        <div className="absolute z-10 mt-1 w-full rounded-md border bg-popover shadow-md">
          {results.map((model) => (
            <button
              key={model.model_id}
              type="button"
              className={cn(
                "w-full px-3 py-2 text-left text-sm hover:bg-accent transition-colors",
                "flex items-center justify-between"
              )}
              onMouseDown={() => {
                setQuery(model.model_id);
                onChange(model.model_id);
                setOpen(false);
              }}
            >
              <span className="font-medium">{model.model_id}</span>
              <span className="text-xs text-muted-foreground">
                {model.architecture ?? ""}
                {model.likes != null && ` \u2665 ${model.likes}`}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
