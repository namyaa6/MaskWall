import { Linter } from "eslint";
import { describe, expect, it } from "vitest";
import { effectReturnRules } from "./effects.js";

const linter = new Linter({ configType: "flat" });
const lint = (code: string) =>
  linter.verify(code, [{ languageOptions: { ecmaVersion: 2022, sourceType: "module" }, rules: { "no-restricted-syntax": ["error", ...effectReturnRules] } }]);

describe("effect return-value rule", () => {
  it.each([
    ["implicit return of a call (the Chat bug)", "useEffect(() => el.current?.scrollIntoView?.({ block: 'end' }), [n]);"],
    ["implicit return of a value", "useEffect(() => count, []);"],
    ["async arrow", "useEffect(async () => { await go(); }, []);"],
    ["async function expression", "useLayoutEffect(async function () { await go(); }, []);"],
  ])("flags: %s", (_n, code) => expect(lint(code)).toHaveLength(1));

  it.each([
    ["block body", "useEffect(() => { el.scrollIntoView(); }, [n]);"],
    ["cleanup function (block)", "useEffect(() => { const id = setInterval(f, 1); return () => clearInterval(id); }, []);"],
    ["cleanup function (expression)", "useEffect(() => () => { w.terminate(); }, []);"],
    ["async work inside a block", "useEffect(() => { (async () => { await go(); })(); }, []);"],
    ["not an effect", "useMemo(() => compute(), []);"],
  ])("allows: %s", (_n, code) => expect(lint(code)).toHaveLength(0));
});
