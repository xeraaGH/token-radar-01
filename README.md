# Launch Radar — Token Radar v2

[![make.fun](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/xeraaGH/token-radar-01/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard)
[![reward](https://make.fun/badge/token-radar-01/xeraaGH/token-radar-01/reward.svg)](https://make.fun/bounty/token-radar-01/leaderboard)

A token launch radar that reads DefiLlama's sealed protocol snapshot, works
out which protocols still record no token, and ranks them by what a launch
would be worth — with a fitted value model, tested out of sample, and a
costed farm plan for every protocol on the board.

## The radar

- **178 pre-token protocols ranked** — all 40 of the snapshot's watchlist,
  plus every protocol below the $50M floor that passes the house's own
  watchlist rule (no token recorded, not a wrapped / staked / bridged
  product, not a versioned pool, not an exchange). The 92 protocols above
  the floor that the house rule leaves out are set aside and named, not
  hidden.
- **A fitted value model, not a thumb rule.** The implied value is
  `TVL x 10^predicted log-ratio`, where the log market-cap/TVL ratio is a
  ridge regression fitted on the 93 tokenised protocols most like the
  board (exchanges, bridges and versioned pools excluded), on log TVL,
  7-day growth, age since listing, and a shrunk category comps term.
  Targets are winsorised; the category term shrinks toward the global
  median by `n / (n + 5)`.
- **Tested twice, reported honestly.** Held-out split (every 5th protocol
  by slug): R² 0.31, median error 93%, 22% of predictions within 2x of the
  real market cap. Walk-forward split (trained on protocols listed before
  2024, tested on those listed after): R² 0.22, median error 84%, 34%
  within 2x. Both beat the median-ratio baseline on R². Every row carries
  a ±1σ value band from the walk-forward error, so no number pretends to
  be precise.
- **A costed farm plan for every row.** Six steps each: the capital to put
  in (from the protocol's own TVL tier), the gas, the weekly cadence, the
  hold time (from its age), and how to claim safely. Each row gets a net
  verdict — FARM (top decile of implied value), THIN (next 40%), SKIP
  (the rest, or any protocol bleeding 40%+ of TVL in a week).
- **The whole universe accounted for.** The integrity strip names all 758
  protocols: ranked, live-token (the model's training set, shown with
  actual vs predicted market cap), and set aside with the reason.

## The dashboard

A static page that renders with no network: the board (sortable,
filterable by category, chain, verdict and watchlist), a detail view for
every protocol (model breakdown, value band, farm plan, verdict), a live
tokens view (actual vs predicted market cap), and the method panel with
the fitted coefficients and both validation tests.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input /path/to/snapshot.json
```

`main.ts` (Node 22, standard library only) reads the snapshot, fits the
model, writes `board.json`, `data.js`, copies `dashboard/` and
`screenshots/` to the output directory, and writes `dossier.json` with
the vendored make.fun library. Two runs write identical bytes.

## Honesty notes

- "Records no token" is DefiLlama's record, never proof a protocol has
  no token.
- An implied value is this entry's own claim about what a launch would be
  worth if the protocol's record stayed as it is. It does not predict
  whether or when a token is issued.
- The model's typical miss is large (about 2x either way at the median);
  the value band on each row says so in plain numbers.
