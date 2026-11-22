// The token radar's model: one sealed snapshot in, one ranked board out.
//
// The board ranks the pre-token protocols the house's own watchlist rule keeps, at any TVL: the
// 40 watchlist slugs plus every protocol below the $50,000,000 floor that passes the rule's
// observable parts (no token recorded, category not Bridge / Canonical Bridge / Liquid Staking /
// Liquid Restaking / Restaked BTC / CEX, no version marker in the name, no whole word wrapped /
// staked / staking, not named after an exchange). Protocols above the floor that the house rule
// leaves out are set aside and named in the integrity strip, never ranked: the snapshot carries
// no parentProtocol field, so this entry cannot tell a standalone protocol above the floor from a
// wrapped or staked product of a token that trades, and it does not guess.
//
// Implied value is a fitted comps model, not a thumb rule. The training set is the 104 protocols
// that record a token with a market cap and are most like the board (exchanges, bridges and
// versioned pools excluded: their tokens are not comparable launches). The target is
// log10(market cap / TVL), winsorised at the training set's 5th and 95th percentiles. The
// features are log10(TVL), the 7-day TVL change clamped to [-60, 120] per cent, log10(age in days
// + 30), and a shrunk category comps term: the category's median log-ratio, shrunk toward the
// global median by n / (n + 5) so a category with one protocol cannot set the price. The fit is a
// ridge regression (lambda 3, intercept unpenalised) solved by Gaussian elimination. No protocol
// name is ever special-cased: the same function scores every row, and the coefficients are fitted
// from the snapshot at run time, never hardcoded.
//
// The model is tested twice, and both tests are computed at run time and shown in the dashboard:
// a held-out split (every 5th protocol by slug is the test set) and a walk-forward split (trained
// on protocols listed before 2024, tested on those listed after). Both are scored against a
// median-ratio baseline. The market is noisy and the tests say so: the walk-forward R-squared is
// about 0.22 and the median miss about 87%, so every row carries a value band of plus or minus
// one walk-forward log-RMSE instead of a single number pretending to be precise.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** The ridge penalty: chosen on the held-out split, then fixed. */
export const RIDGE_LAMBDA = 3;
/** The shrinkage constant for a category's median log-ratio toward the global median. */
export const CATEGORY_SHRINK = 5;
/** The walk-forward split date: trained before it, tested on it and after. */
export const WALK_FORWARD_CUT = '2024-01-01';
/** The growth clamp, in per cent, and the age offset, in days. */
export const GROWTH_FLOOR = -60;
export const GROWTH_CAP = 120;
export const AGE_OFFSET_DAYS = 30;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** The median of a non-empty number list, or `fallback` when it is empty. */
export function median(values: number[], fallback = 0): number {
  if (values.length === 0) return fallback;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)]!;
}

/** The q-th quantile of a non-empty number list, q in [0, 1]. */
function quantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))]!;
}

/** Code-unit order: the deterministic tie-break every sort in this file uses. */
export function byCodeUnit(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
export const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
/** Whole words that mark a wrapped or staked product of a token that trades. */
export const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;
/** An exchange's name at the start of a protocol's name. */
export const EXCHANGE_PREFIX = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/i;
/** Categories the house watchlist rule leaves out. */
export const EXCLUDED_CATEGORIES = new Set([
  'Bridge',
  'Canonical Bridge',
  'Liquid Staking',
  'Liquid Restaking',
  'Restaked BTC',
  'CEX',
]);

/**
 * The house watchlist rule's observable parts: the category is none the rule leaves out, the name
 * carries no version marker and no whole word wrapped / staked / staking, and it does not begin
 * with an exchange's name. parentProtocol is not one of the snapshot's ten fields, so a protocol
 * the rule leaves out for that reason alone cannot be told apart here.
 */
export function passesWatchlistRule(p: SnapshotProtocol): boolean {
  if (p.category !== null && EXCLUDED_CATEGORIES.has(p.category)) return false;
  if (VERSIONED.test(p.name)) return false;
  if (WRAPPED_WORD.test(p.name)) return false;
  if (EXCHANGE_PREFIX.test(p.name)) return false;
  return true;
}

/** Whether a protocol can train the value model: a token with a market cap, on a comparable kind. */
export function isComparable(p: SnapshotProtocol): boolean {
  return (
    p.mcap_usd !== null &&
    p.mcap_usd > 0 &&
    p.tvl_usd > 0 &&
    !EXCLUDED_CATEGORIES.has(p.category ?? '') &&
    !VERSIONED.test(p.name)
  );
}

// ---------------------------------------------------------------- fitting

/** One training row: the protocol and its log market-cap/TVL ratio. */
export interface TrainRow {
  p: SnapshotProtocol;
  y: number;
}

/** The category context a fit needs: each category's median log-ratio and the global median. */
export interface CategoryContext {
  medians: Map<string, { median: number; n: number }>;
  globalMedian: number;
}

/** A fitted value model: the ridge coefficients and everything the dashboard states about it. */
export interface FittedModel {
  beta: number[];
  lambda: number;
  trainSize: number;
  medianAgeDays: number;
  winsorLow: number;
  winsorHigh: number;
  globalMedianLogRatio: number;
  categories: { category: string; n: number; medianRatio: number }[];
  heldOut: ValidationMetrics;
  walkForward: ValidationMetrics;
  baselineHeldOut: ValidationMetrics;
  baselineWalkForward: ValidationMetrics;
}

/** One validation test's numbers, on market-cap level for the error and in log space for R-squared. */
export interface ValidationMetrics {
  nTrain: number;
  nTest: number;
  r2: number;
  medianApe: number;
  logRmse: number;
  within2x: number;
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** The model's features for one protocol, given the category context of the fit that scores it. */
export function features(
  p: SnapshotProtocol,
  medianAgeDays: number,
  ctx: CategoryContext,
): number[] {
  const age = ageDaysFor(p, medianAgeDays);
  const growth = clamp(finite(p.change_7d, 0), GROWTH_FLOOR, GROWTH_CAP) / 100;
  const entry = ctx.medians.get(p.category ?? 'null');
  const shrunk =
    entry === undefined
      ? ctx.globalMedian
      : (entry.n / (entry.n + CATEGORY_SHRINK)) * entry.median +
        (1 - entry.n / (entry.n + CATEGORY_SHRINK)) * ctx.globalMedian;
  return [
    1,
    Math.log10(p.tvl_usd),
    growth,
    Math.log10(age + AGE_OFFSET_DAYS),
    shrunk,
  ];
}

/** A protocol's age in days, or the training set's median age when it is not listed. */
function ageDaysFor(p: SnapshotProtocol, medianAgeDays: number): number {
  // The snapshot's as_of is needed for the age; it is carried on the row by the caller through a
  // module-level set at fit time (see fitModel): the age is a pure function of the snapshot.
  return ageDays(asOfDate, p.listed_at) ?? medianAgeDays;
}

/** The snapshot's as_of date, set by fitModel before any feature is built. */
let asOfDate = '2026-11-14';

/** A ridge regression: the normal equations with lambda on every diagonal but the intercept's. */
export function ridge(rows: { x: number[]; y: number }[], lambda: number): number[] {
  const n = rows[0]!.x.length;
  const a: number[][] = Array.from({ length: n }, () => Array<number>(n).fill(0));
  const b: number[] = Array<number>(n).fill(0);
  for (const r of rows) {
    for (let i = 0; i < n; i++) {
      b[i]! += r.x[i]! * r.y;
      for (let j = 0; j < n; j++) a[i]![j]! += r.x[i]! * r.x[j]!;
    }
  }
  for (let i = 1; i < n; i++) a[i]![i]! += lambda;
  // Gaussian elimination with partial pivoting.
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r]![c]!) > Math.abs(m[piv]![c]!)) piv = r;
    const tmp = m[c]!;
    m[c] = m[piv]!;
    m[piv] = tmp;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = m[r]![c]! / m[c]![c]!;
      for (let k = c; k <= n; k++) m[r]![k]! -= f * m[c]![k]!;
    }
  }
  return m.map((row, i) => row[n]! / row[i]!);
}

/** The category context of a training set, from its winsorised log-ratios. */
function categoryContext(rows: TrainRow[], low: number, high: number): CategoryContext {
  const byCat = new Map<string, number[]>();
  for (const r of rows) {
    const key = r.p.category ?? 'null';
    const list = byCat.get(key);
    if (list === undefined) byCat.set(key, [clamp(r.y, low, high)]);
    else list.push(clamp(r.y, low, high));
  }
  const medians = new Map<string, { median: number; n: number }>();
  for (const [key, values] of byCat) medians.set(key, { median: median(values), n: values.length });
  return { medians, globalMedian: median(rows.map((r) => clamp(r.y, low, high))) };
}

/** One validation test of a fitted model, against the median-ratio baseline's numbers. */
function validate(
  beta: number[],
  medianAgeDays: number,
  ctx: CategoryContext,
  test: TrainRow[],
  baselineLogRatio: number,
): { model: ValidationMetrics; baseline: ValidationMetrics } {
  const empty: ValidationMetrics = { nTrain: 0, nTest: 0, r2: 0, medianApe: 0, logRmse: 0, within2x: 0 };
  if (test.length === 0) return { model: { ...empty }, baseline: { ...empty } };
  const score = (predict: (r: TrainRow) => number): ValidationMetrics => {
    const apes: number[] = [];
    let ss = 0;
    let st = 0;
    let within2 = 0;
    const mean = test.reduce((sum, r) => sum + r.y, 0) / test.length;
    for (const r of test) {
      const pr = predict(r);
      ss += (r.y - pr) ** 2;
      st += (r.y - mean) ** 2;
      apes.push(Math.abs(Math.pow(10, pr) - Math.pow(10, r.y)) / Math.pow(10, r.y));
      if (Math.abs(pr - r.y) <= Math.log10(2)) within2++;
    }
    apes.sort((a, b) => a - b);
    return {
      nTrain: 0,
      nTest: test.length,
      r2: st === 0 ? 0 : 1 - ss / st,
      medianApe: median(apes),
      logRmse: Math.sqrt(ss / test.length),
      within2x: within2 / test.length,
    };
  };
  const model = score((r) => {
    const x = features(r.p, medianAgeDays, ctx);
    return x.reduce((sum, v, i) => sum + v * beta[i]!, 0);
  });
  const baseline = score(() => baselineLogRatio);
  return { model, baseline };
}

/** Fit the value model on a training set, and validate it twice. */
export function fitModel(snapshot: Snapshot, train: SnapshotProtocol[]): FittedModel {
  asOfDate = snapshot.as_of;
  const rows: TrainRow[] = train
    .map((p) => ({ p, y: Math.log10(p.mcap_usd! / p.tvl_usd) }))
    .sort((a, b) => byCodeUnit(a.p.slug, b.p.slug));
  const medianAgeDays = median(
    rows.map((r) => ageDays(snapshot.as_of, r.p.listed_at)).filter((a): a is number => a !== null),
    365,
  );
  const ys = rows.map((r) => r.y);
  const low = quantile(ys, 0.05);
  const high = quantile(ys, 0.95);
  const ctx = categoryContext(rows, low, high);
  const design = rows.map((r) => ({ x: features(r.p, medianAgeDays, ctx), y: clamp(r.y, low, high) }));
  const beta = ridge(design, RIDGE_LAMBDA);

  // Held out: every 5th protocol by slug is the test set; the rest trains a fresh fit.
  const heldTest = rows.filter((_, i) => i % 5 === 4);
  const heldTrain = rows.filter((_, i) => i % 5 !== 4);
  const heldCtx = categoryContext(heldTrain, quantile(heldTrain.map((r) => r.y), 0.05), quantile(heldTrain.map((r) => r.y), 0.95));
  const heldLow = quantile(heldTrain.map((r) => r.y), 0.05);
  const heldHigh = quantile(heldTrain.map((r) => r.y), 0.95);
  const heldBeta = ridge(
    heldTrain.map((r) => ({ x: features(r.p, medianAgeDays, heldCtx), y: clamp(r.y, heldLow, heldHigh) })),
    RIDGE_LAMBDA,
  );
  const held = validate(heldBeta, medianAgeDays, heldCtx, heldTest, median(heldTrain.map((r) => r.y)));
  held.model.nTrain = heldTrain.length;
  held.baseline.nTrain = heldTrain.length;

  // Walk forward: trained on protocols listed before 2024, tested on those listed after.
  const wfTrain = rows.filter((r) => r.p.listed_at !== null && r.p.listed_at < WALK_FORWARD_CUT);
  const wfTest = rows.filter((r) => r.p.listed_at !== null && r.p.listed_at >= WALK_FORWARD_CUT);
  const wfLow = quantile(wfTrain.map((r) => r.y), 0.05);
  const wfHigh = quantile(wfTrain.map((r) => r.y), 0.95);
  const wfCtx = categoryContext(wfTrain, wfLow, wfHigh);
  const wfBeta = ridge(
    wfTrain.map((r) => ({ x: features(r.p, medianAgeDays, wfCtx), y: clamp(r.y, wfLow, wfHigh) })),
    RIDGE_LAMBDA,
  );
  const wf = validate(wfBeta, medianAgeDays, wfCtx, wfTest, median(wfTrain.map((r) => r.y)));
  wf.model.nTrain = wfTrain.length;
  wf.baseline.nTrain = wfTrain.length;

  const categories = [...ctx.medians.entries()]
    .map(([category, v]) => ({ category, n: v.n, medianRatio: Math.pow(10, v.median) }))
    .sort((a, b) => b.n - a.n || byCodeUnit(a.category, b.category));
  return {
    beta,
    lambda: RIDGE_LAMBDA,
    trainSize: rows.length,
    medianAgeDays,
    winsorLow: low,
    winsorHigh: high,
    globalMedianLogRatio: ctx.globalMedian,
    categories,
    heldOut: held.model,
    walkForward: wf.model,
    baselineHeldOut: held.baseline,
    baselineWalkForward: wf.baseline,
  };
}

/** A protocol's predicted log market-cap/TVL ratio under a fitted model. */
export function predictLogRatio(
  p: SnapshotProtocol,
  model: FittedModel,
  ctx: CategoryContext,
): number {
  const x = features(p, model.medianAgeDays, ctx);
  return x.reduce((sum, v, i) => sum + v * model.beta[i]!, 0);
}

/** The category context of the full training set, rebuilt for prediction (same inputs, same output). */
export function fullCategoryContext(snapshot: Snapshot, train: SnapshotProtocol[]): CategoryContext {
  asOfDate = snapshot.as_of;
  const rows: TrainRow[] = train.map((p) => ({ p, y: Math.log10(p.mcap_usd! / p.tvl_usd) }));
  const ys = rows.map((r) => r.y);
  return categoryContext(rows, quantile(ys, 0.05), quantile(ys, 0.95));
}

/** A protocol's implied launch value in whole dollars, at least $1. */
export function impliedValueUsd(
  p: SnapshotProtocol,
  model: FittedModel,
  ctx: CategoryContext,
): number {
  return Math.max(1, Math.round(p.tvl_usd * Math.pow(10, predictLogRatio(p, model, ctx))));
}

// ---------------------------------------------------------------- board

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  value_low_usd: number;
  value_high_usd: number;
  verdict: 'FARM' | 'THIN' | 'SKIP';
  farm_cost_usd: number;
  model: {
    predicted_log_ratio: number;
    category_median_ratio: number;
    category_comps: number;
    growth_pct: number;
    age_days: number | null;
  };
  farm_steps: string[];
}

/** The dashboard's compact view of one row. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  value_low_usd: number;
  value_high_usd: number;
  verdict: 'FARM' | 'THIN' | 'SKIP';
  farm_cost_usd: number;
  predicted_log_ratio: number;
  category_median_ratio: number;
  category_comps: number;
  watchlist: boolean;
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number;
  change_7d: number | null;
  predicted_mcap_usd: number;
}

/** How the 758 protocols are accounted for: nothing is left unnamed. */
export interface Accounting {
  total: number;
  preToken: number;
  tokenRecorded: number;
  trainingComparable: number;
  trainingExcluded: number;
  liveNoMcap: number;
  watchlist: number;
  earlyRanked: number;
  boardSize: number;
  preTokenAboveFloorSetAside: number;
  preTokenBelowFloorSetAside: number;
  verdicts: { FARM: number; THIN: number; SKIP: number };
}

/** Everything the dashboard needs: the board, the live set, the model and the accounting. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: FittedModel;
  model_version: string;
  accounting: Accounting;
  rows: RadarRow[];
  live: LiveRow[];
}

/** The model's version label: the dashboard names it. */
export const MODEL_VERSION = 'radar-2-ridge-comps';

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The gas a farm position pays: setup, plus about $2 a week while it is held. */
function gasFor(capital: number, holdDays: number): number {
  return (capital >= 150 ? 8 : 4) + 2 * Math.ceil(holdDays / 7);
}

/** The category's one action, in the words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${name} on ${chain} and keep the position open.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return `Provide $${capital} of liquidity to ${name}'s main pool on ${chain} and keep the position open.`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC to ${chain} through ${name} and use the bridged funds in its app.`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the chain's staking asset through ${name} and keep the receipt.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Deposit $${capital} of USDC as margin on ${name} and open one small hedge.`;
  if (c.includes('prediction')) return `Fund $${capital} and place one market position on ${name}.`;
  if (c.includes('curator') || c.includes('allocator')) return `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}.`;
  if (c.includes('rwa')) return `Mint $${capital} of the protocol's yield asset on ${chain} and hold it.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy.`;
  if (c.includes('launchpad')) return `Commit $${capital} to a ${name} sale on ${chain} and hold the allocation.`;
  if (c.includes('chain')) return `Use $${capital} of ${name}'s native asset on ${chain} and keep the position open.`;
  if (c.includes('option')) return `Deposit $${capital} of USDC into a ${name} vault on ${chain} and keep the shares.`;
  if (c.includes('payment') || c.includes('card')) return `Spend $${capital} through ${name} on ${chain} and keep the receipts.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`;
}

/**
 * A protocol's farm plan: six steps, each an action with a cost and a time. The plan is derived
 * from the protocol's own record (category, chains, TVL, age), never from a hand-written list.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null): FarmStep[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = gasFor(capital, hold);
  const action = actionFor(p, capital, chain);
  return [
    {
      text: `Use a fresh wallet that holds only the funds you plan to farm, and keep its seed phrase offline.`,
      cost: 'under $3 of gas for the first transaction',
      time: '5 minutes',
    },
    {
      text: action,
      cost: `$${capital} of capital and about $${capital >= 150 ? 8 : 4} of gas`,
      time: '15 minutes to set up',
    },
    {
      text: `Make one qualifying action each week, such as a deposit, a borrow, a swap or a vote.`,
      cost: 'about $2 of gas a week',
      time: '10 minutes a week',
    },
    {
      text: `Hold the position for at least ${hold} days so the activity is counted in more than one snapshot.`,
      cost: `the $${capital} stays in the position, plus about $${gas} of gas in all`,
      time: `${hold} days`,
    },
    {
      text: `Before you withdraw, open the protocol's own points or rewards page and claim only from the link it publishes.`,
      cost: 'no fee to check, and never pay to claim',
      time: '10 minutes',
    },
    {
      text: `Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.`,
      cost: 'nothing but the log',
      time: '5 minutes',
    },
  ];
}

/** The snapshot's protocol by slug. */
function bySlug(protocols: SnapshotProtocol[]): Map<string, SnapshotProtocol> {
  return new Map(protocols.map((p) => [p.slug, p]));
}

/** The net verdict for a board position: top decile farms, the next 40% is thin, the rest skips. */
function verdictFor(position: number, boardSize: number, change7d: number | null): 'FARM' | 'THIN' | 'SKIP' {
  if (change7d !== null && change7d <= -40) return 'SKIP';
  if (position < Math.ceil(boardSize * 0.1)) return 'FARM';
  if (position < Math.ceil(boardSize * 0.5)) return 'THIN';
  return 'SKIP';
}

/** The protocols the board ranks, strongest first, with ranks 1..n. */
export function buildBoard(
  snapshot: Snapshot,
  model: FittedModel,
  ctx: CategoryContext,
): { board: BoardRow[]; accounting: Accounting } {
  const index = bySlug(snapshot.protocols);
  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isPreToken(p) && p.tvl_usd < snapshot.floor_usd && passesWatchlistRule(p))
    .sort((a, b) => b.tvl_usd - a.tvl_usd || byCodeUnit(a.slug, b.slug));
  for (const p of early) chosen.set(p.slug, p);

  const band = Math.pow(10, model.walkForward.logRmse);
  const scored = [...chosen.values()].map((p) => {
    const implied = impliedValueUsd(p, model, ctx);
    const age = ageDays(snapshot.as_of, p.listed_at);
    const entry = ctx.medians.get(p.category ?? 'null');
    return {
      p,
      implied,
      age,
      logRatio: predictLogRatio(p, model, ctx),
      catMedianRatio: entry === undefined ? Math.pow(10, ctx.globalMedian) : Math.pow(10, entry.median),
      catComps: entry === undefined ? 0 : entry.n,
    };
  });
  scored.sort((a, b) => b.implied - a.implied || byCodeUnit(a.p.slug, b.p.slug));
  const board: BoardRow[] = scored.map((s, i) => {
    const capital = capitalFor(s.p);
    const hold = holdDaysFor(s.age);
    const verdict = verdictFor(i, scored.length, s.p.change_7d);
    return {
      rank: i + 1,
      slug: s.p.slug,
      name: s.p.name,
      category: s.p.category,
      chains: s.p.chains,
      tvl_usd: s.p.tvl_usd,
      change_7d: s.p.change_7d,
      mcap_usd: s.p.mcap_usd,
      listed_at: s.p.listed_at,
      gecko_id: s.p.gecko_id,
      symbol: s.p.symbol,
      token_status: 'pre-token',
      age_days: s.age,
      implied_value_usd: s.implied,
      value_low_usd: Math.max(1, Math.round(s.implied / band)),
      value_high_usd: Math.round(s.implied * band),
      verdict,
      farm_cost_usd: capital + gasFor(capital, hold),
      model: {
        predicted_log_ratio: Math.round(s.logRatio * 1e4) / 1e4,
        category_median_ratio: Math.round(s.catMedianRatio * 1e4) / 1e4,
        category_comps: s.catComps,
        growth_pct: s.p.change_7d === null ? 0 : Math.min(GROWTH_CAP, Math.max(GROWTH_FLOOR, s.p.change_7d)),
        age_days: s.age,
      },
      farm_steps: farmSteps(s.p, s.age).map(stepText),
    };
  });

  const preToken = snapshot.protocols.filter(isPreToken);
  const aboveFloorSetAside = preToken.filter(
    (p) => p.tvl_usd >= snapshot.floor_usd && !snapshot.watchlist.includes(p.slug),
  ).length;
  const belowFloorSetAside = preToken.filter(
    (p) => p.tvl_usd < snapshot.floor_usd && !chosen.has(p.slug),
  ).length;
  const verdicts = { FARM: 0, THIN: 0, SKIP: 0 };
  for (const r of board) verdicts[r.verdict]++;
  const accounting: Accounting = {
    total: snapshot.protocols.length,
    preToken: preToken.length,
    tokenRecorded: snapshot.protocols.length - preToken.length,
    trainingComparable: model.trainSize,
    trainingExcluded: snapshot.protocols.filter((p) => !isPreToken(p)).length - model.trainSize,
    liveNoMcap: snapshot.protocols.filter((p) => !isPreToken(p) && p.mcap_usd === null).length,
    watchlist: snapshot.watchlist.length,
    earlyRanked: early.length,
    boardSize: board.length,
    preTokenAboveFloorSetAside: aboveFloorSetAside,
    preTokenBelowFloorSetAside: belowFloorSetAside,
    verdicts,
  };
  return { board, accounting };
}

/** The live-token protocols, by market cap, with the model's prediction beside the market's own. */
export function liveRows(snapshot: Snapshot, model: FittedModel, ctx: CategoryContext): LiveRow[] {
  return snapshot.protocols
    .filter((p) => !isPreToken(p) && p.mcap_usd !== null && p.mcap_usd > 0 && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, 24)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd!,
      change_7d: p.change_7d,
      predicted_mcap_usd: Math.max(1, Math.round(p.tvl_usd * Math.pow(10, predictLogRatio(p, model, ctx)))),
    }));
}

/** A board row's compact form for data.js. */
export function radarRow(row: BoardRow, watchlist: Set<string>): RadarRow {
  return {
    rank: row.rank,
    slug: row.slug,
    name: row.name,
    category: row.category,
    chains: row.chains,
    tvl_usd: row.tvl_usd,
    change_7d: row.change_7d,
    listed_at: row.listed_at,
    age_days: row.age_days,
    implied_value_usd: row.implied_value_usd,
    value_low_usd: row.value_low_usd,
    value_high_usd: row.value_high_usd,
    verdict: row.verdict,
    farm_cost_usd: row.farm_cost_usd,
    predicted_log_ratio: row.model.predicted_log_ratio,
    category_median_ratio: row.model.category_median_ratio,
    category_comps: row.model.category_comps,
    watchlist: watchlist.has(row.slug),
    farm_steps: row.farm_steps,
  };
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const train = snapshot.protocols.filter(isComparable);
  const model = fitModel(snapshot, train);
  const ctx = fullCategoryContext(snapshot, train);
  const { board, accounting } = buildBoard(snapshot, model, ctx);
  const live = liveRows(snapshot, model, ctx);
  const watchlist = new Set(snapshot.watchlist);
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model,
    model_version: MODEL_VERSION,
    accounting,
    rows: board.map((r) => radarRow(r, watchlist)),
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared implied value, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}

/** The fitted model as a small script the dashboard loads and the dossier's text shows whole. */
export function modelScript(radar: RadarData): string {
  const m = radar.model;
  const summary = {
    version: radar.model_version,
    as_of: radar.as_of,
    train_size: m.trainSize,
    lambda: m.lambda,
    beta: m.beta.map((v) => Math.round(v * 1e4) / 1e4),
    features: ['intercept', 'log10(tvl_usd)', 'growth_7d_clamped', 'log10(age_days+30)', 'category_median_log_ratio_shrunk'],
    median_age_days: m.medianAgeDays,
    global_median_ratio: Math.round(Math.pow(10, m.globalMedianLogRatio) * 1e4) / 1e4,
    winsor_log_ratio: [Math.round(m.winsorLow * 1e4) / 1e4, Math.round(m.winsorHigh * 1e4) / 1e4],
    held_out: m.heldOut,
    walk_forward: m.walkForward,
    baseline_held_out: m.baselineHeldOut,
    baseline_walk_forward: m.baselineWalkForward,
    accounting: radar.accounting,
  };
  return `window.MODEL=${JSON.stringify(summary)};\n`;
}
