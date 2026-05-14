# Multi-Stock Platform Upgrade Handoff

Date: 2026-04-09

## Scope summary

This pass was the platform-generalization cut that moves LightningTrader4000 from a mostly QQQ-shaped workstation into a multi-stock advisory platform with honest per-symbol autonomy readiness, per-symbol workspaces, and stock-aware news relevance routing. The chosen implementation deliberately favored a coherent foundation over a giant rewrite:

- Introduce shared instrument-profile infrastructure instead of scattered symbol heuristics.
- Add persistent per-stock workspace state and make the GUI behave like a workspace switcher rather than a single global QQQ page.
- Generalize the news pipeline from a QQQ/tech worldview into a stock-profile-aware relevance system.
- Preserve the recent stock-path truthfulness work by keeping tentative/blocked/watchlist distinctions and by not pretending new symbols are actual-autonomy ready.
- Protect smoothness by using lazy runtime bundles, per-symbol cache invalidation, and bounded workspace overview loading instead of duplicating full pages for every stock.

This is a multi-stock advisory foundation plus shadow-autonomy foundation, not a declaration that arbitrary symbols are production-ready for real autonomous trading.

## A. Platform architecture

### Exact modules and functions inspected

The main inspection target list was:

- `stock_gui.py`
  - `LightningGUI.__init__`
  - `_bootstrap_workspace_state`
  - `_refresh_instrument_choices`
  - `_on_instrument_change`
  - `mode_apply`
  - `_on_profile_change`
  - `_refresh_workspace_tabs`
  - `_store_runtime_bundle`
  - `_runtime_bundle`
  - `_apply_runtime_bundle`
  - `_invalidate_workspace_caches`
  - `_activate_workspace`
  - `_open_selected_workspace`
  - `_close_active_workspace`
  - `_open_overview_workspace`
  - `_open_add_stock_dialog`
  - `_initial_model_waiter`
  - `_train_and_reload`
  - `reload_once_threaded`
  - `retrain_once`
  - `_compute_overview_rows`
  - `background_loop`
- `config.py`
  - default instrument registry assembly
  - external instrument loading
  - default context symbol construction
- `instrument_onboarding.py`
  - `register_instrument`
  - runtime instrument discovery
  - default metadata generation
- `manage_instruments.py`
  - CLI add/update path for symbols
- `news_pipeline.py`
  - `_archive_query`
  - `_headline_event_detail`
  - `_headline_sentiment_detail`
  - `_normalize_news_frame`
  - `_parse_item`
  - `_parse_archive_item`
  - `load_news_features_for_index`
  - `load_recent_headlines`
- `advisory_engine.py`
  - `_instrument_meta`
  - `_news_context_snapshot`
  - `_expression_paths`
- `autonomy_engine.py`
  - `_instrument_meta`
  - `_stock_path_status`
  - `_build_snapshot`
- `trader_analytics.py`
  - `_instrument_meta`
  - `_news_event_tag`
  - news row formatting
- `features.py`
  - comments/feature intent around news context
- Also inspected the surrounding architecture indirectly through existing symbol-scoped calls in:
  - `model_utils.py`
  - `model_trainers.py`
  - `replay_engine.py`
  - `forecast_calibration.py`
  - `readiness_tracker.py`

### What already generalized cleanly before this pass

Several important subsystems were already better than the UI suggested:

- Model artifact loading/training was largely already `instrument_key` driven.
- Replay and readiness logic already had symbol-aware persistence and file naming in several places.
- The workstation had already gained multi-horizon support for stock-only paths and better raw-vs-finished translation.
- The GUI already exposed a symbol selector and used the selected symbol in a number of prediction/advisory calls.

That meant the right move was not a full backend rewrite. The highest-value gap was turning the app from “one page that can point at different symbols” into “a multi-stock workstation with persistent workspaces, profile-aware news, honest readiness, and bounded runtime state.”

### What was still tightly coupled to QQQ

The most important residual QQQ bias before this pass was concentrated in four areas:

1. Instrument metadata defaults
- `config.py`, `instrument_onboarding.py`, and `manage_instruments.py` were still building symbol defaults through ad hoc logic that was effectively QQQ-first.
- Default context symbols and profile traits were not centralized, so new symbols did not start with a serious stock profile.

2. News architecture
- `news_pipeline.py` still described itself and behaved like a QQQ/tech sentiment layer.
- Archive query construction, relevance boosts, and filtering leaned heavily on QQQ/tech assumptions.
- The system had relevance, tech, and political dimensions, but not a stock-profile-aware notion of sector, company specificity, or broad-market vs symbol-specific scope.

3. Advisory/autonomy messaging
- Advisory expression copy still said “Direct QQQ stock” in generic stock-path messaging.
- Autonomy still hard-blocked non-QQQ actual usage with a rollout-specific assumption rather than a reusable validation model.

4. GUI state model
- The selector changed the active symbol, but the UI still behaved like a single workspace with global mutable state.
- There was no persistent “page/workspace” concept with remembered mode/profile/status per stock.
- Cache invalidation was too coarse for multi-stock use.
- Overview loading would have become expensive if many symbols were registered because it could fan out across the whole runtime list.

### What was dangerously ad hoc

- Instrument profile knowledge was duplicated instead of centralized.
- News relevance was shaped by keyword heuristics that knew a lot about tech/QQQ but not enough about other sectors.
- GUI background loaders could overwrite active runtime state if the user switched symbols during a load.
- The active instrument concept existed mostly as `instrument_var`, which is too weak for a scalable workstation model.

### New platform-level infrastructure added

#### 1. `instrument_profile.py`

This file is the new shared metadata spine for symbol onboarding and relevance routing.

Core functions:

- `normalize_symbol`
- `normalize_sector`
- `infer_instrument_kind`
- `default_news_theme_weights`
- `default_context_daily_symbols`
- `build_instrument_meta`
- `summarize_instrument_meta`

What it now owns:

- sector normalization aliases
- ETF vs equity inference
- sector ETF / vol proxy / lead future defaults
- style-profile defaults
- macro/regulatory/broad-market sensitivity defaults
- news theme weights by sector and instrument type
- curated symbol overrides for major ETFs like `QQQ`, `SPY`, `DIA`, `IWM`
- alias generation for symbol/company/news matching

This replaced multiple smaller QQQ-shaped helpers and gives every symbol a richer default profile from the moment it is onboarded.

#### 2. `workspace_state.py`

This is the new persistence layer for stock-specific pages/workspaces.

State file:

- `data/workspace_state.json`

Stored state:

- `active_symbol`
- workspace order
- per-symbol workspace entries with:
  - `symbol`
  - `title`
  - `display_name`
  - `instrument_kind`
  - `sector`
  - `industry`
  - `opened`
  - `pinned`
  - timestamps
  - `view_mode`
  - `profile_name`
  - `last_status`
  - `last_note`

Key functions:

- `load_workspace_state`
- `save_workspace_state`
- `open_workspace`
- `activate_workspace`
- `close_workspace`
- `get_workspace`
- `remember_workspace_view`
- `list_workspaces`
- `list_workspace_symbols`

This is the new answer to “what is the active instrument?” It is no longer just a combobox value; it is a persisted workspace with per-stock UI memory.

### How per-stock workspaces/pages now work

The current implementation is intentionally a lightweight workstation page model:

- There is still one main widget tree.
- The app now treats each selected symbol as a workspace tab rather than a one-off global selection.
- Workspace tabs are rendered in a `Workspace Pages` row under the summary bar.
- Each workspace remembers its mode and trader profile.
- Each workspace tracks last note/status such as `train_needed`, `ready`, `Reloaded`, `Retrained`, or `Using cached workspace context`.
- The overview table can open a workspace directly.
- Operators can add a stock workspace from the GUI, which also registers the instrument profile.

This was chosen because it gets the app to a coherent per-stock page model without the performance cost and maintenance burden of duplicating the full UI into separate notebook tabs or windows.

### How stock-specific state is now stored and routed

There are now three layers of symbol-specific state:

1. Registry/profile layer
- `INSTRUMENTS`
- instrument metadata via `build_instrument_meta`

2. Persistent workspace layer
- `workspace_state.json`
- stores operator-facing workspace memory

3. Runtime hot-state layer inside `LightningGUI`
- `self._workspace_runtime_cache`
- bounded per-symbol loaded bundle of:
  - `models`
  - trimmed `df`
  - `loaded_at`

Routing rules:

- Switching workspaces updates the symbol combobox, mode, and trader profile to that workspace’s saved values.
- Cached runtime state is applied immediately if present.
- Background reload still occurs after switching, but completed loads only overwrite `self.models` and `self.df` if the symbol is still active.
- Prediction/dashboard caches are invalidated per symbol rather than globally.

### How training and artifacts are selected per stock

The codebase already used `instrument_key` in many training/inference paths. This pass focused on making the operator workflow consistently route into those symbol-aware paths.

What changed operationally:

- The Add Stock dialog creates or registers symbol metadata, opens the workspace, and optionally starts training.
- Reload/retrain calls continue to pass the selected symbol into `load_models`, `update_price_data`, and `train_all_models`.
- Workspace status tracks whether a symbol still needs training.
- The UI now honestly reflects “models ready” vs “train needed” in workspace summaries.

Important nuance:

- The architecture is now multi-stock-ready.
- The repository snapshot is not magically populated with trained artifacts for arbitrary symbols.
- New symbols still need real data and training before they are advisory-useful.

## B. News architecture

### Why the old news system was not enough

The previous system had already become more truthful, but it still behaved too much like:

- “Is this headline relevant to QQQ/tech?”
- “Is it political?”

That is not enough for a multi-stock workstation because:

- A symbol-specific Apple headline should matter far more to `AAPL` than to `COST`.
- Consumer-retail headlines should matter more to Costco than to QQQ.
- Broad macro headlines should matter to many symbols, but with different weights.
- ETFs and operating companies should not be treated identically.

### New stock-profile concepts added

Through `instrument_profile.py`, each symbol now carries:

- `instrument_kind`
  - `equity` or `etf`
- `sector`
- `industry`
- `style_profile`
- `macro_sensitivity`
- `regulatory_sensitivity`
- `broad_market_sensitivity`
- `news_theme_weights`
- `news_aliases`
- `context_daily_symbols`

The `news_theme_weights` map is the key profile input. It gives each symbol a relevance prior across themes such as:

- `technology`
- `communications`
- `consumer`
- `healthcare`
- `financials`
- `industrials`
- `energy`
- `materials`
- `utilities`
- `real_estate`
- `macro`
- `regulatory`
- `broad_market`

### New relevance and classification rules

`news_pipeline.py` now computes more structured relevance detail for each headline.

Added conceptual outputs:

- `symbol_specific_relevance`
- `sector_relevance`
- `macro_relevance`
- `broad_market_relevance`
- `relevance_scope`
- `relevance_scope_label`
- `scope_score`
- `theme_bucket`
- `theme_relevance`
- `symbol_specific`
- `sector_specific`
- `macro_relevant`
- `broad_market_relevant`
- `opinionated`

The current scope categories are:

- `symbol_specific`
- `sector_specific`
- `macro_political_regulatory`
- `broad_market`
- `low_value_noise`
- `opinion_commentary`

These classifications are produced heuristically from:

- company/symbol alias hits
- sector-theme keyword sets
- macro/political keyword sets
- source/noise scoring
- instrument-profile theme weights

### Archive query changes

Old behavior was effectively QQQ-special-cased.

New `_archive_query(symbol)` behavior:

- uses company/symbol aliases from `news_aliases`
- mixes in top theme terms derived from the symbol’s profile
- includes sector ETF context when relevant

This broadens the archive fetch from “QQQ-shaped query” to “symbol + profile shaped query.”

### Event scoring changes

`_headline_event_detail` now slightly boosts event significance when:

- the headline is strongly symbol-specific
- the headline is strongly aligned with the dominant theme(s) for that instrument

This is still heuristic, but it is directionally much closer to how a real multi-stock workstation should route catalyst significance.

### News-frame normalization changes

`_normalize_news_frame` now persists the richer relevance/scope structure into the stored news CSVs. That means downstream consumers do not need to recompute everything from scratch conceptually; they inherit a more structured news record.

Additional importance weighting changes:

- importance now incorporates `scope_score`
- symbol-specific stories are favored
- opinion/commentary is penalized

### Feature-generation changes

`load_news_features_for_index` now filters stories using multi-stock relevance logic instead of a QQQ-ish tech/politics filter. Stories are kept if they are:

- symbol-specific
- sector-specific
- macro-relevant
- broad-market-relevant
- tech-relevant
- political-relevant
- or generally relevant above a threshold

This is important because it lets the symbol-specific daily feature stream inherit a more appropriate set of headlines without pretending all symbols should use the same news lens.

### Recent-headline routing changes

`load_recent_headlines` now returns scope and theme fields, and ranking includes `scope_score`. The trader-facing layers can now say not only “this headline was positive/negative” but also “it was symbol-specific,” “sector-specific,” or “macro/political.”

### What still remains crude or heuristic

The news system is improved, but it is still rule-based. Remaining simplifications:

- No learned per-symbol relevance model yet.
- No real issuer/entity resolution beyond alias heuristics.
- Sector/theme mappings are hand-authored.
- Macro sensitivity and theme weights are defaults, not validated econometric estimates.
- Opinion/noise classification is lightweight keyword/source logic.
- Archive query generation is broader but still approximate.

This is acceptable for this pass because the goal was platform generalization and truthful routing, not a full NLP rebuild.

## C. UX / smoothness architecture

### Core decisions made to keep the app smooth

The multi-stock upgrade was implemented with explicit anti-lag constraints:

1. Do not duplicate the full GUI per symbol.
- The app uses one main widget tree.
- Stock workspaces are persistent tabs over a shared workstation surface.

2. Cache warmed symbol runtime bundles.
- `self._workspace_runtime_cache` stores trimmed `df` plus loaded models.
- Bounded size: 2 recent workspaces.
- TTL-based staleness handling.

3. Switch fast, reload in background.
- On workspace activation, if cached runtime exists it is applied immediately.
- The dashboard updates from cache first.
- A background reload then refreshes the symbol.

4. Prevent background loads from clobbering the active page.
- Async workers only write into `self.models`/`self.df` if the symbol being loaded is still the active workspace.
- Otherwise they just update the per-symbol runtime bundle.

5. Invalidate caches surgically.
- Dashboard and prediction payload caches are invalidated by symbol.
- Other symbols remain warm.

6. Bound overview fan-out.
- `_compute_overview_rows` now prioritizes open workspaces and caps the list to 8 symbols.
- It does not blindly spin over every registered instrument on every refresh.

7. Persist operator context.
- view mode and trader profile are remembered per stock
- workspace notes/status give immediate context on whether the stock is warmed, retrained, or still needs training

### User-experience changes

Operator workflow now looks like this:

1. Choose a stock from the combobox or add a new one.
2. Open it as a workspace.
3. The workspace tab appears in `Workspace Pages`.
4. The page remembers its mode and trader profile.
5. The banner shows stock context, profile, and whether models are ready.
6. Switching back to a recent stock is fast because it can reuse cached context before background reload finishes.

This is much closer to a real workstation feel than the old “global symbol selection only” model.

### Risks that still remain on the smoothness side

- The GUI still relies on a single Python/Tkinter process and shared widget tree, so pathological numbers of symbols are not a target yet.
- Overview predictions still perform real work for included workspaces; the cap reduces cost but does not remove it.
- There is still no formal memory-pressure policy beyond the small runtime bundle cache.
- Long-running real-market use needs a soak test for repeated workspace switching, training, auto-refresh, and replay interactions.

## D. Code changes

### Files changed in this pass

#### `instrument_profile.py`
- Category: platform generalization, news-system change
- New file.
- Added centralized instrument profile construction, sector normalization, ETF/equity inference, context symbol defaults, and theme-weight defaults.
- Why: this is the reusable metadata layer needed for onboarding, news relevance, and stock-aware UI summaries.

#### `workspace_state.py`
- Category: platform generalization, UI/workspace change
- New file.
- Added persisted workspace/page state with active symbol, order, per-stock mode/profile/status memory.
- Why: the app needed a real per-stock workspace abstraction instead of just a global selected symbol.

#### `config.py`
- Category: platform generalization
- Refactored default context and instrument meta helpers to use `build_instrument_meta`.
- External instrument loading now normalizes incoming metadata through the shared profile builder.
- Why: remove duplicate QQQ-shaped defaulting logic and make registry entries richer and consistent.

#### `instrument_onboarding.py`
- Category: platform generalization
- Refactored onboarding defaults and registration to use `build_instrument_meta`.
- Registration now normalizes alias lists and writes richer metadata.
- Why: newly onboarded symbols need first-class stock profiles from day one.

#### `manage_instruments.py`
- Category: platform generalization
- Refactored CLI default metadata generation to use `build_instrument_meta`.
- Why: command-line symbol onboarding should match GUI/runtime onboarding behavior.

#### `news_pipeline.py`
- Category: news-system change, platform generalization
- Rewrote parts of the relevance model from QQQ/tech-centric heuristics toward symbol-profile-aware routing.
- Added sector/theme keyword families beyond tech.
- Added scope classification and richer relevance output.
- Generalized archive query generation.
- Propagated scope/theme fields into stored rows and recent-headline outputs.
- Adjusted feature filtering to keep symbol/sector/macro/broad-market relevant stories.
- Why: multi-stock support is incomplete if the news system still thinks like a QQQ-only app.

#### `advisory_engine.py`
- Category: platform generalization, stock-signal preservation
- Refactored instrument metadata access through `build_instrument_meta`.
- Generalized stock expression copy from QQQ-specific to symbol-specific.
- `news_context_snapshot` now carries dominant news scope so the trader-facing summary explains whether the top catalyst is symbol-specific, sector-specific, or macro.
- Why: preserve the improved truthfulness while removing QQQ-only messaging.

#### `autonomy_engine.py`
- Category: platform generalization, UI/workspace change, stock-signal preservation
- Refactored instrument metadata access through `build_instrument_meta`.
- Replaced QQQ-only actual autonomy block with a validation model:
  - validated symbols may use actual modes
  - unvalidated symbols can still use advisory + shadow mode honestly
- Added validation label/detail into the autonomy snapshot rows.
- Generalized stock expression copy.
- Why: every workspace should support the autonomy framework, but only validated symbols should be presented as actual-autonomy capable.

#### `trader_analytics.py`
- Category: platform generalization, news-system change
- Refactored metadata access through `build_instrument_meta`.
- Added scope shorthand into news event tags and rows.
- Why: trader-facing diagnostics should reflect the richer news relevance model.

#### `features.py`
- Category: stock-signal preservation, platform generalization
- Updated comments/intent description around news context to reflect per-stock use instead of QQQ-only framing.
- Why: keep feature-layer documentation aligned with the new architecture.

#### `stock_gui.py`
- Category: UI/workspace change, performance change, platform generalization
- Added workspace bootstrap, workspace tabs, add-stock workflow, per-workspace runtime cache, per-symbol invalidation, bounded overview loading, and async safety for workspace switching.
- Added operator-visible workspace summary/status behavior.
- Why: this is the main workstation transformation from a single-instrument app into a multi-stock platform without building a laggy monster.

#### `tests/test_workspace_state.py`
- Category: validation
- New file.
- Verifies workspace persistence, view/profile memory, and active-symbol fallback on close.

#### `tests/test_news_pipeline.py`
- Category: validation
- Added profile-aware multi-stock news relevance test.
- Confirms the same tech headline scores more relevant for `AAPL` than `COST` and that symbol-specific Apple news scopes correctly.

#### `tests/test_stock_gui.py`
- Category: validation, performance change coverage
- Added per-symbol cache invalidation test.
- Confirms invalidating one workspace does not wipe warmed caches for another.

#### `tests/test_autonomy_engine.py`
- Category: validation, stock-signal preservation
- Added non-validated-symbol shadow-only test.
- Confirms non-validated symbols remain usable in shadow mode while actual autonomy is still blocked.

### Files that were already dirty before this pass and were not the focus of this upgrade

The working tree already contained other modified/untracked files, including:

- `execution_engine.py`
- `model_utils.py`
- `model_trainers.py`
- `forecast_calibration.py`
- `catalyst_providers.py`
- several other tests

Those existing changes were not reverted. Future work should review them in context, but they were not the architectural center of this multi-stock pass.

## E. Validation details

### Syntax validation

Ran:

- `python3 -m py_compile instrument_profile.py workspace_state.py config.py instrument_onboarding.py manage_instruments.py news_pipeline.py advisory_engine.py autonomy_engine.py trader_analytics.py features.py stock_gui.py`

Result:

- passed

### Targeted tests run

Ran individually:

- `python3 -m unittest discover -s tests -p 'test_workspace_state.py'`
- `python3 -m unittest discover -s tests -p 'test_news_pipeline.py'`
- `python3 -m unittest discover -s tests -p 'test_stock_gui.py'`
- `python3 -m unittest discover -s tests -p 'test_instrument_onboarding.py'`
- `python3 -m unittest discover -s tests -p 'test_advisory_engine.py'`
- `python3 -m unittest discover -s tests -p 'test_autonomy_engine.py'`
- `python3 -m unittest discover -s tests -p 'test_trader_analytics.py'`

Result:

- all passed

### Full suite run

Ran:

- `python3 -m unittest discover -s tests -p 'test_*.py'`

Observed result:

- `Ran 149 tests in 8.594s`
- `OK`

### Validation coverage achieved

Explicitly verified:

- stock selection / workspace creation logic
- workspace-state persistence and isolation
- per-symbol cache invalidation behavior
- symbol-profile-aware news relevance logic
- shadow-only autonomy handling for unvalidated symbols
- preserved existing advisory/autonomy/trader analytics test coverage
- full-suite regression safety across existing tests

### What was only partially validated

- Real GUI interaction was validated through code paths and unit tests, not through long manual desktop soak sessions.
- Real-time repeated workspace switching under live refresh/replay/training concurrency still needs soak testing.
- Multi-symbol training/artifact routing is validated at code path level, but not yet with a large bench of newly onboarded real symbols.
- News relevance behavior is validated via targeted tests, but the heuristic weights still need practical review across sectors.

### Environmental caveats

- Some test runs emitted expected remote-data warnings from yfinance-related paths in the sandboxed environment.
- The suite still passed; these warnings were informational rather than failures.

## F. Remaining weaknesses

### Residual architecture issues

1. `INSTRUMENTS` still seeds only `QQQ` by default in `config.py`.
- This is acceptable as a bootstrap symbol, but it is still a symbolic single-name default.

2. Legacy fallback variables still exist.
- `SYMBOL`
- `DEFAULT_INSTRUMENT`
- a number of deeper modules still rely on these as fallback behavior
- They do not block multi-stock operation, but they remain residual single-symbol bias.

3. The UI workspace model is lightweight, not full multi-view virtualization.
- This was intentional for smoothness.
- It is the right tradeoff for now, but it means “workspace page” is a persistent state layer over a shared workstation, not separate full page objects.

### Residual QQQ-centric assumptions

- Actual autonomy rollout still defaults to `AUTONOMY_SUPPORTED_SYMBOLS=QQQ`.
- Many local artifact stores in the repo are still QQQ-populated in practice.
- Some deeper comments, defaults, or operational expectations in untouched modules may still assume QQQ as the most common instrument.

### Signal limitations that this pass does not solve

- It does not magically fix the unresolved QQQ finished-stock edge problem.
- It does not prove that any arbitrary newly-added stock has strong deployable signal quality.
- It does not expand into options autonomy.

### News-system shortcuts taken

- Relevance remains heuristic and profile-weighted rather than learned.
- Scope assignment is threshold-based.
- Alias matching is not a full entity-linking system.

### Performance risks still worth watching

- If many workspaces are opened and constantly refreshed, overview computation and context reloads can still become noticeable.
- Runtime bundle cache size is small by design; this protects memory but may reduce warm-switch benefits if the operator cycles through many symbols.

## G. Next-pass advice for ChatGPT

### What to inspect first next time

Start with these files/modules:

- `stock_gui.py`
  - confirm workspace switching remains race-safe under real use
  - inspect any remaining global assumptions around `instrument_var`
- `news_pipeline.py`
  - review relevance thresholds and theme weights with real headline samples across several sectors
- `autonomy_engine.py`
  - inspect how validation list expansion should work once additional symbols are ready
- `advisory_engine.py`
  - inspect whether trader-facing stock-path translation still needs tightening for tentative vs blocked vs watchlist language
- `model_utils.py`, `model_trainers.py`, `forecast_calibration.py`
  - focus on symbol-specific edge quality and readiness, not platform scaffolding

### What not to waste time revisiting

- Do not rebuild the workspace system into a giant multi-tab widget duplication unless there is a concrete usability or performance bug.
- Do not re-litigate whether the app should stay single-symbol. That platform decision is now implemented.
- Do not spend the next pass rewriting onboarding metadata from scratch; `instrument_profile.py` is the intended extension point.
- Do not expand into options autonomy during the next logic-focused phase unless the task explicitly changes.

### Recommended next smaller logic-focused passes

1. Improve symbol-specific training/readiness calibration.
- Compare post-train readiness behavior across a small curated set such as `QQQ`, `AAPL`, `MSFT`, `COST`, `JPM`, `XOM`.
- Tighten per-horizon deployability thresholds where false readiness appears.

2. Attack the unresolved QQQ finished-stock edge directly.
- Use the new platform only as scaffolding.
- Focus on label quality, cost-aware edge retention, horizon blending, and calibration rather than more UI work.

3. Add smarter symbol defaults where needed.
- Curate a few more major-name overrides in `instrument_profile.py` if specific symbols show obviously poor default sector/style/news behavior.

4. Improve news weighting with observed outcomes.
- Consider sector-specific or symbol-specific empirical tuning of theme weights.
- Consider learning adaptive scope weights from realized next-day or next-session reaction.

5. Add more explicit workspace readiness/status surfacing.
- For example: “advisory-ready,” “shadow-only,” “models missing,” “history thin,” “news degraded.”

6. Soak-test the workstation.
- Repeated workspace switching
- reload/retrain during switching
- replay while another workspace is warm
- longer auto-refresh sessions

### Practical decision boundary after this pass

Treat the platform as:

- ready for multi-stock advisory workflows
- ready for multi-stock shadow-autonomy framework workflows
- not yet evidence that many non-QQQ symbols are real-autonomy candidates

That distinction matters. The large architectural lift is now in place, so the next passes should be much narrower and more empirical.
