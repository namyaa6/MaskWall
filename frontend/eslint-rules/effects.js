// An effect callback must return nothing or a cleanup function. Returning anything else (an async function's Promise, or the
// value of an expression body like `() => el.scrollIntoView()`) makes React call it as a cleanup function and crash with
// "destroy is not a function", and only in browsers where that value is not undefined, which is why it can hide from tests.
// Shared by eslint.config.js and eslint-rules/effects.test.ts.
const EFFECT = "CallExpression[callee.name=/^use(Layout)?Effect$/]";
const MESSAGE = "An effect must return nothing or a cleanup function. Use a block body `{ ... }` and, for async work, call an async function inside it.";

export const effectReturnRules = [
  { selector: `${EFFECT} > :matches(ArrowFunctionExpression, FunctionExpression)[async=true]:first-child`, message: `Async effect callback. ${MESSAGE}` },
  {
    selector: `${EFFECT} > ArrowFunctionExpression[expression=true]:first-child:not([body.type=/^(ArrowFunctionExpression|FunctionExpression)$/])`,
    message: `Effect callback implicitly returns a value. ${MESSAGE}`,
  },
];
