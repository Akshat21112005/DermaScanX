"use client";

import { useMemo, useState } from "react";
import { AlertTriangle, UploadCloud, Activity, Stethoscope, Loader2 } from "lucide-react";

type PredictionResponse = {
  prediction?: {
    class?: string;
    predicted_class?: string;
    confidence?: number;
    probabilities?: Record<string, number>;
    all_probabilities?: Record<string, number>;
  };
  predicted_class?: string;
  confidence?: number;
  all_probabilities?: Record<string, number>;
  analysis?: string | Record<string, string>;
  medical_analysis?: string;
  llm_advice?: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://jiten-333-skin-cancer.hf.space";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const prediction = useMemo(() => {
    if (!result) return null;
    const pred = result.prediction;
    return {
      className: pred?.class || pred?.predicted_class || result.predicted_class || "Unknown",
      confidence: pred?.confidence ?? result.confidence ?? 0,
      probabilities: pred?.probabilities || pred?.all_probabilities || result.all_probabilities || {}
    };
  }, [result]);

  const analysisText = useMemo(() => {
    if (!result) return "";
    if (typeof result.analysis === "string") return result.analysis;
    if (result.analysis && typeof result.analysis === "object") {
      return Object.entries(result.analysis)
        .filter(([, value]) => value)
        .map(([key, value]) => `${titleCase(key)}:\n${value}`)
        .join("\n\n");
    }
    return result.medical_analysis || result.llm_advice || "";
  }, [result]);

  function titleCase(value: string) {
    return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function handleFileChange(selectedFile: File | null) {
    setResult(null);
    setError("");
    setFile(selectedFile);

    if (preview) URL.revokeObjectURL(preview);
    setPreview(selectedFile ? URL.createObjectURL(selectedFile) : null);
  }

  async function handleSubmit() {
    if (!file) {
      setError("Please upload a skin image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = (await response.json()) as PredictionResponse;
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong while predicting.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 px-4 py-8">
      <section className="mx-auto max-w-6xl">
        <div className="mb-8 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-medium text-blue-700 shadow-sm">
            <Stethoscope size={18} /> AI Medical Awareness Tool
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-slate-950 md:text-5xl">
            Skin Disease Detection
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-slate-600">
            Upload a skin image. The backend model predicts the class, then the LLM explains the problem, future impact, solution, and advice.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded-3xl border border-white/70 bg-white/90 p-6 shadow-xl backdrop-blur">
            <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900">
              <UploadCloud className="text-blue-600" /> Upload Image
            </h2>

            <label className="flex min-h-[280px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-blue-200 bg-blue-50/60 p-6 text-center transition hover:border-blue-400 hover:bg-blue-50">
              {preview ? (
                <img src={preview} alt="Uploaded preview" className="max-h-64 rounded-xl object-contain" />
              ) : (
                <>
                  <UploadCloud size={52} className="mb-4 text-blue-500" />
                  <p className="text-lg font-semibold text-slate-800">Choose skin image</p>
                  <p className="mt-2 text-sm text-slate-500">PNG, JPG, JPEG supported</p>
                </>
              )}
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
              />
            </label>

            {file && (
              <p className="mt-3 truncate text-sm text-slate-600">
                Selected: <span className="font-medium">{file.name}</span>
              </p>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-2xl bg-blue-600 px-5 py-3 font-semibold text-white shadow-lg transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-300"
            >
              {loading ? <Loader2 className="animate-spin" /> : <Activity />}
              {loading ? "Analyzing..." : "Predict & Generate Report"}
            </button>

            {error && (
              <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {error}
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="rounded-3xl border border-white/70 bg-white/90 p-6 shadow-xl backdrop-blur">
              <h2 className="mb-4 text-xl font-semibold text-slate-900">Prediction Result</h2>

              {prediction ? (
                <div className="space-y-5">
                  <div className="rounded-2xl bg-slate-950 p-5 text-white">
                    <p className="text-sm text-slate-300">Predicted Class</p>
                    <h3 className="mt-1 text-3xl font-bold">{prediction.className}</h3>
                    <p className="mt-3 text-sm text-slate-300">Confidence</p>
                    <div className="mt-2 h-3 overflow-hidden rounded-full bg-slate-700">
                      <div
                        className="h-full rounded-full bg-blue-400"
                        style={{ width: `${Math.min(100, Math.max(0, prediction.confidence))}%` }}
                      />
                    </div>
                    <p className="mt-2 text-lg font-semibold">{prediction.confidence}%</p>
                  </div>

                  <div>
                    <h4 className="mb-3 font-semibold text-slate-800">Class Probabilities</h4>
                    <div className="space-y-3">
                      {Object.entries(prediction.probabilities).map(([name, value]) => (
                        <div key={name}>
                          <div className="mb-1 flex justify-between gap-4 text-sm">
                            <span className="font-medium text-slate-700">{name}</span>
                            <span className="text-slate-500">{value}%</span>
                          </div>
                          <div className="h-2 overflow-hidden rounded-full bg-slate-200">
                            <div
                              className="h-full rounded-full bg-blue-600"
                              style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="rounded-2xl bg-slate-50 p-8 text-center text-slate-500">
                  Result yaha show hoga after image upload.
                </p>
              )}
            </div>

            <div className="rounded-3xl border border-white/70 bg-white/90 p-6 shadow-xl backdrop-blur">
              <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900">
                <AlertTriangle className="text-amber-500" /> Medical Awareness Report
              </h2>
              {analysisText ? (
                <pre className="whitespace-pre-wrap rounded-2xl bg-amber-50 p-5 font-sans text-sm leading-7 text-slate-800">
                  {analysisText}
                </pre>
              ) : (
                <p className="rounded-2xl bg-slate-50 p-8 text-center text-slate-500">
                  Problem description, future impact, solution, and advice yaha show hoga.
                </p>
              )}
            </div>
          </div>
        </div>

        <p className="mt-8 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          Disclaimer: This tool is for educational and awareness purposes only. It is not a final diagnosis. Please consult a dermatologist for medical confirmation.
        </p>
      </section>
    </main>
  );
}
