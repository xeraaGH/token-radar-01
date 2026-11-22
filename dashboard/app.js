// Launch Radar's page: renders window.RADAR (the board, from data.js) and window.MODEL (the
// fitted model, from model.js). No network, no framework: every number on the page comes from
// the snapshot through the model in lib/model.ts, and the page only sorts, filters and shows it.
(function () {
  var R = window.RADAR;
  var M = window.MODEL;
  if (!R || !M) {
    document.body.innerHTML = '<p style="padding:20px">Launch Radar could not load its data.</p>';
    return;
  }

  // ---------------------------------------------------------------- helpers
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function money(v) {
    if (v == null) return '—';
    if (v >= 1e9) return '$' + (v / 1e9).toFixed(2) + 'B';
    if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return '$' + Math.round(v / 1e3) + 'K';
    return '$' + Math.round(v);
  }
  function moneyFull(v) {
    if (v == null) return '—';
    if (v >= 1e9) return '$' + (v / 1e9).toFixed(2) + 'B';
    if (v >= 1e6) return '$' + (v / 1e6).toFixed(2) + 'M';
    if (v >= 1e3) return '$' + (v / 1e3).toFixed(1) + 'K';
    return '$' + Math.round(v);
  }
  function pct(v) {
    if (v == null) return '—';
    var cls = v > 0 ? 'pos' : v < 0 ? 'neg' : '';
    return '<span class="' + cls + '">' + (v > 0 ? '+' : '') + v.toFixed(1) + '%</span>';
  }
  function age(v) {
    if (v == null) return '—';
    if (v >= 365) return (v / 365).toFixed(1) + 'y';
    return v + 'd';
  }
  function num(n) {
    return n.toLocaleString('en-US');
  }
  function $(id) {
    return document.getElementById(id);
  }

  // ---------------------------------------------------------------- header facts
  $('meta').innerHTML =
    '<b>Snapshot ' + esc(R.as_of) + '</b> · DefiLlama protocols ≥ $5M TVL<br>' +
    'Watchlist floor ' + money(R.floor_usd) + ' · ' + num(R.accounting.total) + ' protocols';
  $('source-line').textContent =
    'Source: ' + R.source + '. Snapshot frozen ' + R.as_of +
    '; the board is computed from it at run time and never fetched live.';

  // ---------------------------------------------------------------- the model, as fitted
  var beta = M.beta;
  function r2(v) {
    var x = Math.round(v * 100) / 100;
    return (x === 0 ? 0 : x).toFixed(2);
  }
  function signedTerm(coef, label, first) {
    var sign = coef < 0 ? ' − ' : first ? '' : ' + ';
    return sign + Math.abs(coef).toFixed(2) + '·' + label;
  }
  $('model-formula').innerHTML =
    'Implied value = TVL × 10<sup>log-ratio</sup>, where log-ratio is a ridge fit (λ = ' + M.lambda +
    ') on ' + num(M.train_size) + ' tokenised protocols: ' +
    '<span class="big">' + money(R.rows[0].implied_value_usd) + '</span> tops the board.';
  $('model-coefs').textContent =
    'log-ratio = ' + beta[0].toFixed(2) +
    signedTerm(beta[1], 'log10(TVL)') + signedTerm(beta[2], 'growth') +
    signedTerm(beta[3], 'log10(age+30)') + signedTerm(beta[4], 'category comps') +
    '. Median category ratio ' +
    M.global_median_ratio.toFixed(2) + '× TVL; median training age ' +
    num(M.median_age_days) + ' days.';
  function testLine(t, base) {
    return '<span class="big">R² ' + r2(t.r2) + '</span> · median miss ' +
      Math.round(t.medianApe * 100) + '% · ' + Math.round(t.within2x * 100) +
      '% within 2× of the real market cap (n = ' + t.nTest + ', baseline R² ' +
      r2(base.r2) + ', median miss ' + Math.round(base.medianApe * 100) + '%)';
  }
  $('held-out').innerHTML = testLine(M.held_out, M.baseline_held_out);
  $('walk-forward').innerHTML = testLine(M.walk_forward, M.baseline_walk_forward);

  // ---------------------------------------------------------------- integrity strip
  var A = R.accounting;
  $('integrity').innerHTML =
    '<b>' + num(A.total) + ' protocols accounted for:</b> ' +
    '<b>' + num(A.boardSize) + ' ranked</b> (' + num(A.watchlist) + ' watchlist + ' +
    num(A.earlyRanked) + ' early below the floor) · ' +
    num(A.trainingComparable) + ' live tokens trained the model · ' +
    num(A.liveNoMcap) + ' live tokens with no market cap recorded · ' +
    num(A.preTokenAboveFloorSetAside) + ' pre-token above the floor set aside (wrapped / staked / bridged products) · ' +
    num(A.preTokenBelowFloorSetAside) + ' pre-token below the floor set aside (exchanges, versioned pools, same products)';

  // ---------------------------------------------------------------- KPIs
  var top10 = R.rows.slice(0, 10).reduce(function (s, r) { return s + r.implied_value_usd; }, 0);
  var med = R.rows[Math.floor(R.rows.length / 2)].implied_value_usd;
  $('kpis').innerHTML =
    '<div class="kpi"><div class="v">' + num(A.boardSize) + '</div><div class="l">Protocols ranked</div></div>' +
    '<div class="kpi"><div class="v">' + num(A.watchlist) + ' / ' + num(A.watchlist) + '</div><div class="l">Watchlist covered</div></div>' +
    '<div class="kpi"><div class="v">' + money(R.rows[0].implied_value_usd) + '</div><div class="l">Top implied value</div></div>' +
    '<div class="kpi"><div class="v">' + money(med) + '</div><div class="l">Median implied value</div></div>' +
    '<div class="kpi"><div class="v">' + money(top10) + '</div><div class="l">Top 10 combined</div></div>' +
    '<div class="kpi"><div class="v">' + A.verdicts.FARM + ' / ' + A.verdicts.THIN + ' / ' + A.verdicts.SKIP +
    '</div><div class="l">FARM / THIN / SKIP</div></div>';

  // ---------------------------------------------------------------- filters
  var cats = {};
  var chains = {};
  R.rows.forEach(function (r) {
    cats[r.category || 'Uncategorised'] = (cats[r.category || 'Uncategorised'] || 0) + 1;
    r.chains.forEach(function (c) { chains[c] = (chains[c] || 0) + 1; });
  });
  Object.keys(cats).sort(function (a, b) { return cats[b] - cats[a]; }).forEach(function (c) {
    var o = document.createElement('option');
    o.value = c;
    o.textContent = c + ' (' + cats[c] + ')';
    $('f-cat').appendChild(o);
  });
  Object.keys(chains).sort(function (a, b) { return chains[b] - chains[a]; }).slice(0, 40).forEach(function (c) {
    var o = document.createElement('option');
    o.value = c;
    o.textContent = c + ' (' + chains[c] + ')';
    $('f-chain').appendChild(o);
  });

  var state = { q: '', cat: '', chain: '', verdict: '', watch: false, sort: 'rank' };
  function filtered() {
    var rows = R.rows.filter(function (r) {
      if (state.q) {
        var q = state.q.toLowerCase();
        if (r.name.toLowerCase().indexOf(q) < 0 && r.slug.toLowerCase().indexOf(q) < 0) return false;
      }
      if (state.cat && (r.category || 'Uncategorised') !== state.cat) return false;
      if (state.chain && r.chains.indexOf(state.chain) < 0) return false;
      if (state.verdict && r.verdict !== state.verdict) return false;
      if (state.watch && !r.watchlist) return false;
      return true;
    });
    var s = state.sort;
    rows.sort(function (a, b) {
      if (s === 'tvl') return b.tvl_usd - a.tvl_usd || a.rank - b.rank;
      if (s === 'growth') return (b.change_7d == null ? -1e9 : b.change_7d) - (a.change_7d == null ? -1e9 : a.change_7d) || a.rank - b.rank;
      if (s === 'age') return (b.age_days == null ? -1 : b.age_days) - (a.age_days == null ? -1 : a.age_days) || a.rank - b.rank;
      if (s === 'cost') return a.farm_cost_usd - b.farm_cost_usd || a.rank - b.rank;
      return a.rank - b.rank;
    });
    return rows;
  }

  function renderRows() {
    var rows = filtered();
    var html = rows.map(function (r) {
      return '<tr data-slug="' + esc(r.slug) + '">' +
        '<td>' + r.rank + '</td>' +
        '<td><span class="name">' + esc(r.name) + '</span>' +
        (r.watchlist ? '<span class="badge wl">WATCHLIST</span>' : '') +
        '<br><span class="slug">' + esc(r.slug) + '</span></td>' +
        '<td>' + esc(r.category || '—') + '</td>' +
        '<td>' + esc(r.chains.slice(0, 2).join(', ')) + (r.chains.length > 2 ? ' +' + (r.chains.length - 2) : '') + '</td>' +
        '<td class="num">' + money(r.tvl_usd) + '</td>' +
        '<td class="num">' + pct(r.change_7d) + '</td>' +
        '<td class="num">' + age(r.age_days) + '</td>' +
        '<td class="num"><b>' + money(r.implied_value_usd) + '</b><br><span class="slug">' +
        money(r.value_low_usd) + ' – ' + money(r.value_high_usd) + '</span></td>' +
        '<td><span class="verdict ' + r.verdict + '">' + r.verdict + '</span></td>' +
        '<td class="num">$' + num(r.farm_cost_usd) + '</td></tr>';
    }).join('');
    $('rows').innerHTML = html || '<tr><td colspan="10">No protocol matches these filters.</td></tr>';
    $('count').textContent = 'Showing ' + rows.length + ' of ' + R.rows.length;
  }

  // ---------------------------------------------------------------- detail view
  function verdictWhy(r) {
    if (r.change_7d != null && r.change_7d <= -40) {
      return 'SKIP — TVL fell ' + r.change_7d.toFixed(1) + '% in a week, so the capital is leaving whatever the value.';
    }
    if (r.verdict === 'FARM') {
      return 'FARM — top decile of implied value on this board (' + money(r.implied_value_usd) +
        '). The position is worth the gas and the weeks.';
    }
    if (r.verdict === 'THIN') {
      return 'THIN — mid-board value (' + money(r.implied_value_usd) +
        '). Farm it with a small position, or farm the FARM rows first.';
    }
    return 'SKIP — bottom half of implied value on this board (' + money(r.implied_value_usd) +
      '). The expected share of a launch does not pay for the attention.';
  }
  function showDetail(slug, scroll) {
    var r = null;
    for (var i = 0; i < R.rows.length; i++) if (R.rows[i].slug === slug) { r = R.rows[i]; break; }
    if (!r) return;
    var pool = r.implied_value_usd * 0.05;
    var farmersCovered = Math.max(1, Math.round(pool / Math.max(1, r.farm_cost_usd)));
    // The category term the fit actually used: the raw median, shrunk toward the global median.
    var w = r.category_comps / (r.category_comps + 5);
    var shrunkRatio = Math.pow(10, w * Math.log10(Math.max(1e-6, r.category_median_ratio)) + (1 - w) * Math.log10(Math.max(1e-6, M.global_median_ratio)));
    var steps = (r.farm_steps || []).map(function (s) {
      var parts = s.split(' Cost: ');
      var head = parts[0];
      var tail = parts.length > 1 ? parts[1] : '';
      return '<li>' + esc(head) + (tail ? '<span class="ct">Cost: ' + esc(tail) + '</span>' : '') + '</li>';
    }).join('');
    $('detail').hidden = false;
    $('detail').innerHTML =
      '<h2>#' + r.rank + ' ' + esc(r.name) + ' <span class="slug">' + esc(r.slug) + '</span> ' +
      '<span class="verdict ' + r.verdict + '">' + r.verdict + '</span>' +
      (r.watchlist ? '<span class="badge wl">WATCHLIST</span>' : '') + '</h2>' +
      '<p class="sub">' + esc(r.category || 'Uncategorised') + ' · ' + esc(r.chains.join(', ') || 'no chain listed') +
      ' · TVL ' + money(r.tvl_usd) + ' · 7d ' + pct(r.change_7d) + ' · listed ' +
      esc(r.listed_at || 'unknown') + ' (' + age(r.age_days) + ') · records no token</p>' +
      '<div class="dgrid"><div><h3>Model breakdown</h3><table class="kv">' +
      '<tr><td>Implied launch value</td><td><b>' + moneyFull(r.implied_value_usd) + '</b></td></tr>' +
      '<tr><td>Value band (±1 walk-forward σ)</td><td>' + moneyFull(r.value_low_usd) + ' – ' + moneyFull(r.value_high_usd) + '</td></tr>' +
      '<tr><td>Predicted log10(mcap / TVL)</td><td>' + r.predicted_log_ratio.toFixed(2) + '</td></tr>' +
      '<tr><td>Category comps term</td><td>' + shrunkRatio.toFixed(2) + '× effective (median ' +
      r.category_median_ratio.toFixed(2) + '× from ' + r.category_comps + ' protocol' +
      (r.category_comps === 1 ? '' : 's') + ', shrunk toward global ' + M.global_median_ratio.toFixed(2) + '×)</td></tr>' +
      '<tr><td>7-day TVL growth</td><td>' + (r.change_7d == null ? 'not reported' : r.change_7d.toFixed(1) + '%') + '</td></tr>' +
      '<tr><td>Farm cost, all in</td><td>$' + num(r.farm_cost_usd) + '</td></tr>' +
      '</table>' +
      '<p class="verdictline">' + esc(verdictWhy(r)) + '<br>Break-even: all-in cost $' + num(r.farm_cost_usd) +
      '; if 5% of supply went to farmers (an assumption, not a fact) the pool would be ' +
      money(pool) + ' — enough to cover the cost of about ' + num(farmersCovered) + ' farmers like you.</p></div>' +
      '<div><h3>Farm plan — 6 steps, costed</h3><ol class="steps">' + steps + '</ol></div></div>';
    var trs = document.querySelectorAll('#rows tr');
    for (var j = 0; j < trs.length; j++) trs[j].classList.toggle('sel', trs[j].getAttribute('data-slug') === slug);
    if (scroll) {
      // A deep link to one protocol shows its detail view, not the whole board above it.
      document.body.classList.add('deep');
      window.scrollTo(0, 0);
    }
  }

  // ---------------------------------------------------------------- live tokens
  $('live-rows').innerHTML = R.live.map(function (l) {
    var ratio = l.mcap_usd / Math.max(1, l.predicted_mcap_usd);
    return '<tr><td><span class="name">' + esc(l.name) + '</span><br><span class="slug">' + esc(l.slug) + '</span></td>' +
      '<td>' + esc(l.category || '—') + '</td>' +
      '<td class="num">' + money(l.tvl_usd) + '</td>' +
      '<td class="num"><b>' + money(l.mcap_usd) + '</b></td>' +
      '<td class="num">' + money(l.predicted_mcap_usd) + '</td>' +
      '<td class="num">' + ratio.toFixed(2) + '×</td></tr>';
  }).join('');

  // ---------------------------------------------------------------- wiring
  $('q').addEventListener('input', function (e) { state.q = e.target.value; renderRows(); });
  $('f-cat').addEventListener('change', function (e) { state.cat = e.target.value; renderRows(); });
  $('f-chain').addEventListener('change', function (e) { state.chain = e.target.value; renderRows(); });
  $('f-verdict').addEventListener('change', function (e) { state.verdict = e.target.value; renderRows(); });
  $('f-watch').addEventListener('change', function (e) { state.watch = e.target.checked; renderRows(); });
  $('sort').addEventListener('change', function (e) { state.sort = e.target.value; renderRows(); });
  $('rows').addEventListener('click', function (e) {
    var tr = e.target.closest('tr[data-slug]');
    if (tr) {
      showDetail(tr.getAttribute('data-slug'), false);
      try { history.replaceState(null, '', '#detail=' + tr.getAttribute('data-slug')); } catch (err) { /* file: URLs */ }
    }
  });

  // Hash states, for deep links and the screenshots: #detail=<slug>, #view=farm|watchlist|live.
  renderRows();
  var hash = (location.hash || '').slice(1);
  var params = {};
  hash.split('&').forEach(function (part) {
    var kv = part.split('=');
    if (kv[0]) params[kv[0]] = decodeURIComponent(kv[1] || '');
  });
  if (params.view === 'farm') {
    state.verdict = 'FARM';
    $('f-verdict').value = 'FARM';
    renderRows();
  } else if (params.view === 'watchlist') {
    state.watch = true;
    $('f-watch').checked = true;
    renderRows();
  }
  if (params.detail) showDetail(params.detail, true);
  if (params.view === 'live') {
    document.querySelector('.live').scrollIntoView({ behavior: 'auto', block: 'start' });
  }
})();
