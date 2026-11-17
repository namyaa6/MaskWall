import { useCallback, useEffect, useRef, useState } from "react";
import type { FromWorker, ModelInfo, ToWorker } from "./protocol";

export type ModelStatus =
  | { state: "idle" }
  | { state: "loading"; loaded: number; total: number }
  | { state: "ready"; info: ModelInfo }
  | { state: "error"; message: string };

/**
 * One model worker per hook instance, created lazily: nothing is downloaded until `load()` or `run()` is called.
 * `run` resolves with the worker's result; requests are matched by id.
 */
export function useModelWorker<R>(factory: () => Worker) {
  const workerRef = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, { resolve: (r: R) => void; reject: (e: Error) => void }>());
  const nextId = useRef(1);
  const [status, setStatus] = useState<ModelStatus>({ state: "idle" });

  const ensure = useCallback((): Worker => {
    if (workerRef.current) return workerRef.current;
    const w = factory();
    w.onmessage = (ev: MessageEvent<FromWorker<R>>) => {
      const m = ev.data;
      if (m.type === "progress") setStatus({ state: "loading", loaded: m.loaded, total: m.total });
      else if (m.type === "ready") setStatus({ state: "ready", info: m.info });
      else if (m.type === "result") {
        pending.current.get(m.id)?.resolve(m.result);
        pending.current.delete(m.id);
      } else if (m.type === "error") {
        if (m.id !== undefined) {
          pending.current.get(m.id)?.reject(new Error(m.message));
          pending.current.delete(m.id);
        } else setStatus({ state: "error", message: m.message });
      }
    };
    w.onerror = (e) => setStatus({ state: "error", message: e.message || "worker failed" });
    workerRef.current = w;
    return w;
  }, [factory]);

  const load = useCallback(() => {
    setStatus((s) => (s.state === "idle" || s.state === "error" ? { state: "loading", loaded: 0, total: 0 } : s));
    ensure().postMessage({ type: "load" } satisfies ToWorker);
  }, [ensure]);

  const run = useCallback(
    (text: string): Promise<R> => {
      load();
      const id = nextId.current++;
      return new Promise<R>((resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        ensure().postMessage({ type: "run", id, text } satisfies ToWorker);
      });
    },
    [ensure, load],
  );

  useEffect(
    () => () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    },
    [],
  );

  return { status, load, run };
}

export const createPrecheckWorker = () => new Worker(new URL("./precheck.worker.ts", import.meta.url), { type: "module" });
export const createPrivacyWorker = () => new Worker(new URL("./privacy.worker.ts", import.meta.url), { type: "module" });
