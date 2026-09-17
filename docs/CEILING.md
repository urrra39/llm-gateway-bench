# Ceiling: what this benchmark cannot weigh

Conclusion: the workload is constructed, two operating points of one model are
not a market, and a single judge pair with no human verification is not ground
truth. Every cost number is a function of the duplicate fractions below, not a
prediction about anyone's traffic.

1. The workload is built from PAWS paraphrase pairs with fixed duplicate
   fractions (low: 10% exact, 15% paraphrase, 5% trap, 70% novel; high: 30%
   exact, 30% paraphrase, 5% trap, 35% novel). The cache hit rate is a property
   of that construction.
2. Since 2026-09-17 both tiers are deepseek-v4-flash with different prompts and
   token budgets. Router savings are measured token savings, not a comparison
   of vendors.
3. Quality is a stronger-model judge at temperature 0 on a frozen rubric, with
   a second judge from the same model. No human has verified any label.
4. Prices are an assumed per-token rate with the date recorded, not a provider
   quote. The measured quantity is tokens.
5. Latency is measured on a warm process; the first-request cost is reported
   separately and excluded from percentiles. The tail is dominated by upstream
   model variance at measurement time, so p99 gaps between configurations
   partly reflect when each run happened, not only what it ran.
