"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { postJSON } from "@/lib/api";
import type { JobSubmissionRequest, JobSubmissionResponse, ApiError } from "@/lib/types";

const DATASETS = ["MNIST", "Fashion-MNIST", "synthetic"] as const;
const MODEL_TYPES = ["MLP"] as const;

interface FormState {
  job_name: string;
  dataset_name: string;
  model_type: string;
  shard_count: string;
  learning_rate: string;
  epochs: string;
  batch_size: string;
  hidden_layers: string;
}

const DEFAULT_FORM: FormState = {
  job_name: "",
  dataset_name: "MNIST",
  model_type: "MLP",
  shard_count: "1",
  learning_rate: "0.001",
  epochs: "10",
  batch_size: "32",
  hidden_layers: "128,64",
};

function parseHiddenLayers(value: string): number[] {
  return value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => !isNaN(n) && n > 0);
}

function formatApiError(err: unknown): string {
  if (err instanceof Error) {
    const apiErr = err as Error & { data?: ApiError };
    if (apiErr.data) {
      const d = apiErr.data;
      if (typeof d.detail === "string") return d.detail;
      if (Array.isArray(d.detail)) {
        return d.detail.map((e) => `${e.loc?.join(".")}: ${e.msg}`).join("; ");
      }
      if (d.error) return d.error;
    }
    return err.message;
  }
  return "An unexpected error occurred.";
}

interface FieldProps {
  label: string;
  htmlFor: string;
  hint?: string;
  required?: boolean;
  children: React.ReactNode;
}

function Field({ label, htmlFor, hint, required, children }: FieldProps) {
  return (
    <div>
      <label htmlFor={htmlFor} className="block text-sm font-medium text-gray-700">
        {label}
        {required && <span className="ml-1 text-red-500">*</span>}
      </label>
      {hint && <p className="mt-0.5 text-xs text-gray-500">{hint}</p>}
      <div className="mt-1">{children}</div>
    </div>
  );
}

const inputClass =
  "block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500";

export default function NewJobPage() {
  const router = useRouter();
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleChange(
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    const shardCount = parseInt(form.shard_count, 10);
    if (isNaN(shardCount) || shardCount < 1) {
      setError("Worker count must be a positive integer.");
      return;
    }

    const hiddenLayers = parseHiddenLayers(form.hidden_layers);
    if (form.hidden_layers.trim() && hiddenLayers.length === 0) {
      setError("Hidden layers must be comma-separated positive integers (e.g. 128,64).");
      return;
    }

    const payload: JobSubmissionRequest = {
      dataset_name: form.dataset_name,
      model_type: form.model_type,
      shard_count: shardCount,
      hyperparameters: {
        learning_rate: parseFloat(form.learning_rate),
        epochs: parseInt(form.epochs, 10),
        batch_size: parseInt(form.batch_size, 10),
        hidden_layers: hiddenLayers.length > 0 ? hiddenLayers : [128, 64],
      },
    };

    if (form.job_name.trim()) {
      payload.job_name = form.job_name.trim();
    }

    setSubmitting(true);
    try {
      const result = await postJSON<JobSubmissionRequest, JobSubmissionResponse>(
        "/api/jobs",
        payload
      );
      router.push(`/jobs/${result.job_id}`);
    } catch (err) {
      setError(formatApiError(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Submit New Job</h1>
          <p className="mt-1 text-sm text-gray-500">
            Configure and submit a distributed ML training job.
          </p>
        </div>
        <Link href="/jobs" className="text-sm text-gray-500 hover:text-gray-700">
          ← Back to Jobs
        </Link>
      </div>

      <form
        onSubmit={handleSubmit}
        className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm space-y-6"
      >
        {/* Job name */}
        <Field label="Job Name" htmlFor="job_name" hint="Optional human-readable name">
          <input
            id="job_name"
            name="job_name"
            type="text"
            value={form.job_name}
            onChange={handleChange}
            placeholder="e.g. mnist-baseline-run-1"
            className={inputClass}
          />
        </Field>

        {/* Dataset */}
        <Field label="Dataset" htmlFor="dataset_name" required>
          <select
            id="dataset_name"
            name="dataset_name"
            value={form.dataset_name}
            onChange={handleChange}
            className={inputClass}
            required
          >
            {DATASETS.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </Field>

        {/* Model type */}
        <Field label="Model Type" htmlFor="model_type" required>
          <select
            id="model_type"
            name="model_type"
            value={form.model_type}
            onChange={handleChange}
            className={inputClass}
            required
          >
            {MODEL_TYPES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>

        {/* Worker count */}
        <Field
          label="Worker Count"
          htmlFor="shard_count"
          hint="Number of workers for collaborative training (must not exceed idle node count)"
          required
        >
          <input
            id="shard_count"
            name="shard_count"
            type="number"
            min={1}
            step={1}
            value={form.shard_count}
            onChange={handleChange}
            className={inputClass}
            required
          />
        </Field>

        {/* Hyperparameters section */}
        <div className="border-t border-gray-100 pt-4">
          <h2 className="mb-4 text-sm font-semibold text-gray-700">Hyperparameters</h2>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Learning Rate" htmlFor="learning_rate" required>
              <input
                id="learning_rate"
                name="learning_rate"
                type="number"
                step="any"
                min="0.000001"
                value={form.learning_rate}
                onChange={handleChange}
                className={inputClass}
                required
              />
            </Field>

            <Field label="Epochs" htmlFor="epochs" required>
              <input
                id="epochs"
                name="epochs"
                type="number"
                min={1}
                step={1}
                value={form.epochs}
                onChange={handleChange}
                className={inputClass}
                required
              />
            </Field>

            <Field label="Batch Size" htmlFor="batch_size" required>
              <input
                id="batch_size"
                name="batch_size"
                type="number"
                min={1}
                step={1}
                value={form.batch_size}
                onChange={handleChange}
                className={inputClass}
                required
              />
            </Field>

            <Field
              label="Hidden Layers"
              htmlFor="hidden_layers"
              hint="Comma-separated integers, e.g. 128,64"
              required
            >
              <input
                id="hidden_layers"
                name="hidden_layers"
                type="text"
                value={form.hidden_layers}
                onChange={handleChange}
                placeholder="128,64"
                className={inputClass}
                required
              />
            </Field>
          </div>
        </div>

        {/* Error display */}
        {error && (
          <div className="rounded-md bg-red-50 border border-red-200 p-4 text-sm text-red-700">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Submit */}
        <div className="flex items-center justify-end gap-3 border-t border-gray-100 pt-4">
          <Link
            href="/jobs"
            className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            Cancel
          </Link>
          <button
            type="submit"
            disabled={submitting}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? "Submitting…" : "Submit Job"}
          </button>
        </div>
      </form>
    </div>
  );
}
