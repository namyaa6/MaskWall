// npm run export-results: regenerate src/data/results.json from ml/results (see exportResults.ts).
import { writeResults } from "./exportResults";

console.log(`wrote ${writeResults()}`);
