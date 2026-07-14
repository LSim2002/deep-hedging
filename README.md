# Deep Hedging

Two experiments in learning hedging policies directly from simulated paths, each isolating one claim.

**Part 1** — under transaction costs, a CVaR-minimizing FFN beats Black–Scholes delta hedging on a European call (GBM).
**Part 2** — for a path-dependent payoff under stochastic volatility, path signatures supply the memory a Markovian FFN lacks, matching an LSTM at a fraction of the parameters and training time.

*Following Bühler et al. (2018) and Abi Jaber & Gérard (2025).*

---

## Part 1 — FFN vs Black–Scholes under transaction costs

**Setup.** European call, GBM underlying. FFN policy takes `(moneyness, time-to-maturity, previous hedge)` and outputs a hedge ratio; trained end-to-end to minimize CVaR₅% of terminal P&L, with proportional cost `c` on turnover.

**The point.** Black–Scholes delta is optimal for the *frictionless* problem and makes no allowance for the cost of rebalancing. Feeding the previous position back in gives the network the state it needs to trade hedging accuracy against turnover — so once `c` is large enough, it should win, and the advantage should widen with `c`. That is the whole claim; nothing here is about model capacity or efficiency.

<img width="587" height="437" alt="image" src="https://github.com/user-attachments/assets/5fe66a54-d6c8-4773-905c-e5d675b8a7fc" />

<img width="1489" height="710" alt="image" src="https://github.com/user-attachments/assets/57e07ba2-87d2-4d59-90a0-4e14d6bb6e9a" />

**Result.** At `c = 0` the FFN converges to the B&S hedge from above, matching its CVaR to within ~5% — the sanity check that the network approaches a known optimum but cannot beat it. As `c` grows, B&S bleeds cost while the FFN learns to rebalance less: the two cross near `c ≈ 0.01`, and by `c = 0.05` the FFN leads by 0.14 in CVaR.

**On the crossover.** The ~5% residual at `c = 0` is optimization error, not a structural handicap — CVaR₅% only sees 5% of each batch, so convergence is slow. That residual pushes the crossover out to `c ≈ 1%`. More epochs and a larger network would close it; we kept the training budget modest, since the effect of interest is the *slope* — B&S degrades faster in `c` — and that is already unambiguous.

---

## Part 2 — FFN vs Signature NN vs LSTM (Asian, Heston)

**Setup.** Geometric Asian call (ATM, K=1) under Heston. Quadratic hedging loss. Three policies compared against a vol-blind Black–Scholes benchmark that hedges with constant σ = √θ:

| | State |
|---|---|
| **Vanilla NN** | Markovian: `(S_t, t)` only |
| **Signature NN** | `(S_t, t)` + truncated path signature of the path so far |
| **Recurrent NN** | LSTM over the path |
| **Vol-blind B&S** | Naive benchmark: hedges with a constant `σ = √θ` (Heston's long-run level), ignoring the stochastic variance entirely  |

**The point.** The payoff depends on the running average, and volatility is stochastic — so the optimal hedge is *not* a function of the current spot alone. A Markovian FFN is structurally unable to represent it. The question is whether a fixed, hand-computed path summary (the signature) recovers what a learned recurrent state does.

<img width="770" height="470" alt="image" src="https://github.com/user-attachments/assets/039d1707-90cf-42d6-9fdd-0231927f21b4" />
<img width="770" height="470" alt="image" src="https://github.com/user-attachments/assets/118c083b-7341-4cff-b20b-2edf5abb7dc9" />

**Result.** It does. The Vanilla NN plateaus visibly above the other two; the Signature NN and the Recurrent NN converge to roughly the same OOS loss, both edging past the naive BS benchmark. The signature version reaches it with far fewer parameters and less training time — the memory is supplied by the path signature rather than learned.

<img width="690" height="440" alt="image" src="https://github.com/user-attachments/assets/936fdbd2-9f53-4b32-bc38-53da4635235e" />

The P&L density makes the failure mode concrete: the Vanilla NN's distribution is materially wider, while the Signature and Recurrent hedges tighten the tails relative to the vol-blind benchmark.

<img width="1139" height="632" alt="image" src="https://github.com/user-attachments/assets/004089d7-4f31-4718-91c1-3a3668b7925d" />
<img width="899" height="850" alt="image" src="https://github.com/user-attachments/assets/cb7849d1-45e9-42a1-9d24-0b4d4b4292f8" />
<img width="701" height="449" alt="image" src="https://github.com/user-attachments/assets/b82b7124-414e-4d70-9d1f-218928cedda3" />

The difference surface shows *where* the learned hedge departs from vol-blind BS — systematically under-hedging below the strike early in the life of the option, which is the correction stochastic volatility calls for and which a constant-σ delta cannot express.

---

## Note on part 2's results

- Part 2's SNN ≈ RNN result is a statement about **convergence**, not superiority. Under a fixed training budget the signature model gets there faster; given enough epochs the LSTM catches up. The defensible claim is compute/data efficiency, not raw accuracy.

---

## References

- Bühler, Gonon, Teichmann, Wood — *Deep Hedging* (2018)
- Abi Jaber, Gérard — *Hedging with Memory* (2025)
