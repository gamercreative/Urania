# Urania — Project State Summary

> Handoff document. Snapshot of the repo as of commit `42ae2f8` ("wrote api contract with tree"), 2026-08-31.
> Purpose: give a planning agent enough context to produce a work plan without re-reading the whole tree.

---

## 1. What this project is

**Urania** is a machine learning library written **from scratch in Julia** — no Flux, no MLJ, no ScikitLearn wrapper. Every algorithm is implemented from the mathematics up.

Stated design conviction (from `readme.md`): **model, loss, and optimizer are orthogonal.** A model describes a hypothesis, a loss describes what "wrong" means, an optimizer describes how to improve, and none of them should need to know the internals of the others. The same optimizer should be able to drive a closed-form linear regression and an iterative logistic model without special-casing.

Working method the author has committed to:
- Build the **slow naive version first**, treat it as ground truth for later optimized versions.
- **Test each component before moving to the next** — every method is gated by a correctness test.

Repo: `git@github.com:gamercreative/Urania.git`, branch `main`, working tree clean.
Julia 1.11.2. ~620 lines of source across 19 files.

---

## 2. Directory layout

```
Project.toml            # [deps] only — Revise, Statistics, StatsBase, Test
Manifest.toml
readme.md               # philosophy + "things I'm learning as a library builder"
src/
  Urania.jl             # module entry point; ordered include list
  models/
    model.jl            # abstract type hierarchy (the interface spine)
    classic/
      trees/
        definitions.jl  # TreeNode, TreeModel, TreeSpecifications, Leaf, Branch, DecisionTreeClassifier
        tree.jl         # the actual algorithms (gini, best_split, build, traverse)
        format.jl       # public API surface: fit / predict
      forests/
        definitions.jl  # ForestModel, RandomForest, is_fitted
        forest.jl       # create_random_forest, traverse_forest
        format.jl       # currently just re-includes (no fit/predict yet)
      linear_regressions/
        linear.jl       # LinearRegression struct only — no algorithm yet
    deep_learning/
      backprop.jl       # ORPHANED — not included by Urania.jl
  core/
    activation.jl       # Activation abstract + Sigmoid
    loss.jl             # Loss abstract + MSE
    optimizer.jl        # Optimzer abstract + GD (struct only, no step function)
    regularization.jl   # empty — docstring only
    training.jl         # ClassicFit stub with empty body
  data/
    data_preperation.jl # bootstrap_sampling
  tests/
    datasets.jl         # XOR + a 16-row/8-row handmade 4-feature classification set
    models.jl           # tree + forest tests (run at module load)
    data.jl             # bootstrap smoke test (commented out of Urania.jl)
```

**Architectural convention that has emerged** (worth preserving, it is consistent for trees and forests):
each model family is a directory of three files —
`definitions.jl` (types), `<family>.jl` (internal algorithms), `format.jl` (the outward-facing `fit`/`predict` contract).

---

## 3. Type hierarchy (the interface spine)

```
Model (abstract)
├── ClassicModel        # closed-form / one-shot
│   ├── TreeModel
│   │   ├── TreeSpecifications      # config object: max_depth, min_smaple_split [sic]
│   │   └── DecisionTreeClassifier  # max_depth, min_sample_split, root
│   ├── ForestModel
│   │   └── RandomForest            # max_depth, min_sample_split, trees::Vector{TreeModel}
│   └── LinearRegression            # Weights::Matrix, bias — struct only
└── DeepLearningModel   # iterative; declared, nothing implements it yet

ModelType (abstract)
├── Classification      # declared, currently unused
└── Regression          # declared, currently unused

TreeNode (abstract)
├── Leaf{L}             # prediction::L
└── Branch{L}           # feature::Int, threshold::Float64, left, right

Activation → Sigmoid
Loss       → MSE
Optimzer   → GD         # note: typo in the abstract type name
```

---

## 4. What actually works today (verified)

I ran `julia --project=. -e 'include("src/Urania.jl")'`. Dependencies had to be installed first
(`Pkg.instantiate()`); after that, **all 44 assertions pass**:

| Test set | Pass |
|---|---|
| handmade XOR tree test (manual `Leaf`/`Branch` construction + traversal) | 4/4 |
| automatic XOR tree test (`build_classifier_tree_node` learns XOR) | 4/4 |
| random forest construction (10 trees, all fitted, correct types) | 30/30 |
| random forest prediction (2 samples from the handmade test set) | 2/2 |
| tree prediction via the `fit` → `predict` API | 4/4 |

### Decision tree classifier — complete
`src/models/classic/trees/tree.jl` has a full, working CART-style classifier:
- `gini(y)` — impurity, computed via sort + sliding window (O(n log n)) rather than a dict.
- `majority(y)` — most common label via a count dict + `argmax`.
- `split_score(x, y, threshold)` — weighted child gini.
- `calculate_endpoints(x)` — candidate thresholds as midpoints between sorted unique values.
- `best_split(X, y)` — exhaustive scan over every feature × every candidate threshold; returns `nothing` if no split exists.
- `build_classifier_tree_node(...)` — recursive build, stops on `max_depth`, `min_sample_split`, or pure node.
- `build_classifier_tree_model(...)` — wraps the root in a `DecisionTreeClassifier`.
- `traverse_tree(root, x)` — iterative descent (`<=` goes left), errors if it lands on a non-leaf.

### Random forest — naive version complete
`src/models/classic/forests/forest.jl`:
- `create_random_forest(X, y, tree_count, max_depth, min_sample_split)` — bootstrap-samples per tree, builds each tree, wraps in `RandomForest`.
- `traverse_forest(forest, x)` — majority vote via `StatsBase.mode`.
- `is_fitted(forest)` — false if empty or if any tree is unfitted.

**Note: this is bagging, not yet a true random forest.** There is no per-split random feature subsampling (`max_features` / `mtry`); every tree sees all features and only the rows differ.

### Public API contract — started at the last commit
`src/models/classic/trees/format.jl` established the intended outward-facing pattern:
```julia
fit(specs::TreeSpecifications, X, y) -> DecisionTreeClassifier
predict(tree::TreeModel, x::AbstractVector) -> prediction
```
i.e. **a config struct goes in, a fitted model struct comes out.** This is the pattern to replicate for every other model family.

### Data
`bootstrap_sampling(X, y, n_samples)` — samples rows with replacement via `selectdim`. Convention documented in the file: **batch dimension is always dimension 1** (`size(X, 1)`).

---

## 5. What is declared but not implemented

| Item | File | State |
|---|---|---|
| `fit`/`predict` for forests | `forests/format.jl` | File contains only two `include` lines. No `ForestSpecifications`, no `fit`, no `predict`. `PredictedForestWorkflow()` in the tests is an empty function body. |
| Linear regression | `linear_regressions/linear.jl` | `LinearRegression` struct only — no normal equation, no gradient fit, no predict. |
| `ClassicFit` | `core/training.jl` | `function ClassicFit(model::ClassicModel, loss::Loss, )` — empty body, trailing comma in the signature. This is the intended generic training entry point. |
| `DeepFit` | `core/training.jl` | Mentioned in the docstring, does not exist. Meant to use an autograd engine. |
| Optimizer step | `core/optimizer.jl` | `GD(η)` struct exists; there is no `step!`/`update!` function, so nothing consumes an optimizer yet. |
| Regularization | `core/regularization.jl` | Docstring only. Intent stated: regularization should be **its own entity**, not embedded into loss or model. |
| Autograd | — | Referenced repeatedly (`loss.jl`, `training.jl`) as the thing deep learning will depend on. Does not exist anywhere. |
| Backprop | `deep_learning/backprop.jl` | `LinearBackPropagation` is written and looks correct in shape, but **the file is never included by `Urania.jl`**, and its own `include("loss.jl")` / `include("activation.jl")` are wrong relative paths (those files live in `core/`). It is dead code that would fail if wired in as-is. A `CrossEntropy`/`SoftMax` specialization is sketched in a comment. |
| `ModelType` (`Classification`/`Regression`) | `models/model.jl` | Declared but never attached to any model. The dispatch axis exists on paper only. |
| Regression trees | — | Only the classifier exists; `Regression` as a tree variant is unbuilt. |
| Loss library | `core/loss.jl` | Only `MSE`. No CrossEntropy/BCE despite backprop referencing it. |
| Activation library | `core/activation.jl` | Only `Sigmoid`. No SoftMax/ReLU/Tanh despite backprop referencing SoftMax. |

---

## 6. Known defects and rough edges

Ordered roughly by how much they'll bite.

1. **`MSE` is mathematically wrong.** `core/loss.jl:16` reads `(::MSE)(y, ŷ) = mean(ŷ .- y) .^ 2` — that is *(mean of the errors) squared*, not the mean of the squared errors. Errors cancel; the loss reads ~0 for a symmetric error distribution. Should be `mean((ŷ .- y) .^ 2)`. Nothing consumes it yet, so no test caught it.

2. **`Project.toml` has no `name`, `uuid`, or `version`.** It is a bare `[deps]` environment, not a package. `using Urania` does **not** work; the only way to load the code is `include("src/Urania.jl")`. This blocks `Pkg.test`, blocks Revise working properly, and blocks anyone else from depending on it.

3. **No `export` statements anywhere.** Even once it becomes a package, every symbol would need `Urania.` qualification.

4. **Tests run at module load time.** `Urania.jl:34` includes `tests/models.jl`, whose bottom-level statements call the test functions directly. So importing the library *runs the test suite*. There is no `test/runtests.jl`. `tests/data.jl` is commented out at line 35 — the standard "uncomment to run" pattern, which won't survive growth.

5. **Test data lives in the shipped module.** `tests/datasets.jl` is included unconditionally, so `x_xor`, `X_hand_train`, etc. are part of the library's namespace.

6. **Double include of `forests/definitions.jl`.** `forests/format.jl` includes both `definitions.jl` and `forest.jl`, and `forest.jl` includes `definitions.jl` again. It happens to load today, but it's a latent hazard and it makes the include graph ambiguous — compare `trees/format.jl`, which does it the clean way (only `format.jl` includes anything).

7. **`TreeSpecifications <: TreeModel` is a modelling error.** It is a *config* object, but it sits in the *model* branch of the hierarchy, which means `is_fitted(tree::TreeModel)` (`trees/definitions.jl:66`, `!isnothing(tree.root)`) will throw on it — a `TreeSpecifications` has no `root` field. Specs should be their own hierarchy (e.g. `abstract type ModelSpec end`).

8. **Typos baked into the API.** `TreeSpecifications.min_smaple_split` (in the struct, so it's a public field name), `abstract type Optimzer`, `data_preperation.jl`, "Unaria"/"macihne" in the readme's first line. Rename now while there are no downstream users.

9. **Untyped struct fields.** `TreeSpecifications.max_depth`, `.min_smaple_split`, `DecisionTreeClassifier.max_depth`, `.min_sample_split`, and both `RandomForest` config fields have no type annotations — they're `Any`, which kills type stability and is inconsistent with the well-typed `Leaf{L}`/`Branch{L}`.

10. **`fit` signature is over-narrow for a classifier.** `trees/format.jl:10` requires `y::AbstractVector{<:Real}`, but a classifier should accept `String`/`Symbol`/categorical labels — `Leaf{L}` is already generic over label type, and `build_classifier_tree_model` handles any label type fine. The API is stricter than the algorithm.

11. **`DecisionTreeClassifier`'s docstring documents a `fitted` field that does not exist** (`is_fitted` derives it from `root` instead). Stale doc.

12. **`gini` requires sortable labels.** The sort-based implementation is fast but silently constrains label types to those supporting `isless`. Fine for now, worth a comment or a dict fallback.

13. **`.vscode/settings.json` points `julia.environmentPath` at `/Users/akram/Documents/code/julia_ml`** — a different directory than this project. The editor is resolving a different environment than the one the code runs in.

14. **`Revise` and `Test` are listed as regular `[deps]`.** They belong in `[extras]`/`[targets]` (Test) or the global dev environment (Revise), not as hard dependencies of the library.

15. **Forest prediction is only spot-checked.** `TestRandomForest` asserts on samples 1 and 7 of `X_hand_test` — the two "easy" cases. The trap cases (2 and 6, where feature x2 lies) and the boundary cases (4 and 8) are deliberately in the dataset but never asserted on. No accuracy-threshold test exists anywhere.

---

## 7. Conventions to follow when adding code

Inferred from the existing code; a plan should respect these rather than fight them.

- **Three-file pattern per model family**: `definitions.jl` (types) / `<name>.jl` (algorithms) / `format.jl` (public `fit`/`predict`, and the only file that includes the other two).
- **Multiple dispatch over flags** — noted explicitly in the header comments of `tree.jl` and `forest.jl`.
- **Callable structs for math objects**: losses/activations are singleton structs with a call method plus a `gradient(::T, args...)` method (`(::MSE)(y, ŷ)` + `gradient(::MSE, y, ŷ)`). Follow this for every new loss and activation.
- **Broadcasting everywhere** so scalar and array shapes both work (stated in `loss.jl`'s docstring).
- **Batch dimension is dimension 1.**
- **Immutable structs by default** — `tree.jl`'s header notes tree nodes never change after construction. `LinearRegression` is the one `mutable struct` (learnable params).
- **Dense inline commenting**, lowercase, explaining *why* — the author comments almost every block. Docstrings on structs describe each field.
- **Unicode math identifiers** are used freely (`η`, `ŷ`, `∂L_∂w`, `δ`).
- **Tests are functions named `TestXxx`**, wrapping `@testset` blocks, called at the bottom of the file.
- Equations go **in the code, not in the comments** (stated in `loss.jl`).

---

## 8. Trajectory — where the author was heading

Reading the commit history plus the stubs, the momentum is:

1. ✅ Infrastructure + type hierarchy → ✅ decision tree classifier (tested) → ✅ naive random forest (tested) → ✅ started the `fit`/`predict` API contract with trees.
2. **Immediate next step, explicitly half-done:** replicate the `fit`/`predict` contract for forests (`forests/format.jl` is empty, `PredictedForestWorkflow()` is an empty test stub waiting to be filled).
3. Then presumably: make the classic side generic (`ClassicFit` + optimizer step + a real linear/logistic regression), which is where the model/loss/optimizer orthogonality claim in the readme actually gets tested.
4. Longer arc: autograd → wire in `backprop.jl` → deep learning models under the already-declared `DeepLearningModel` branch.

---

## 9. Open design questions a plan should resolve

- Should `TreeSpecifications` become a general `ModelSpec`/`Hyperparameters` hierarchy separate from `Model`? Every model family will need a config object, and the current placement is already broken.
- Where does `ModelType` (`Classification`/`Regression`) attach — as a type parameter (`DecisionTree{Classification}`), a field, or a separate concrete type per task? This decision blocks regression trees.
- Does `fit` return a new model (current tree behaviour) or mutate in place? Trees return new; `LinearRegression` is `mutable`, implying mutation. Pick one before writing `ClassicFit`.
- What is the `ClassicFit(model, loss, optimizer, ...)` signature, and how does a closed-form solver (normal equation) fit the same interface as an iterative one? This is the central claim of the readme and is currently untested by any real code.
- Does `predict` take a single vector (current) or a whole matrix? Batch prediction doesn't exist yet for either trees or forests.
- Is the autograd engine going to be tape-based or operator-overloading-with-dual-numbers? Everything on the deep learning side is blocked behind it.

---

## 10. How to run it

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'include("src/Urania.jl")'
```

The second command loads the module *and* runs the test suite as a side effect (see defect #4).
