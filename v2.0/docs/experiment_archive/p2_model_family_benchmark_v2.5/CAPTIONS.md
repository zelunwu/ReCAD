# Figure captions

## fig01_family_taxonomy.png

Figure 1. The frozen benchmark contains 14 model-family columns spanning linear, bagged-tree, boosted-tree, tabular-neural, and spatiotemporal-neural approaches. SOM and GRaCER are excluded from the columns because they are spatial-organization rows.

## fig02_search_budgets.png

Figure 2. Every family receives a preregistered minimum and maximum optimization budget. A family may stop after the minimum only when the search and capacity tails meet the same empirical-convergence threshold; reaching the maximum without convergence is preserved as optimization_incomplete.

## fig03_seed_requirements.png

Figure 3. Tree and linear families require at least three completed seeds, while neural families require five. Seed counts are convergence requirements and cannot be reduced after outer-fold results are inspected.

## fig04_protocol_weights.png

Figure 4. The frozen primary score gives the largest weight to whole-region transfer, then buffered geography, with complete-cruise, forward-time, and early-era evidence retained. Every protocol uses cruise-equal RMSE normalized by its matched strict control.

## fig05_fidelity_schedule.png

Figure 5. Every family follows the same 120,000-row, 300,000-row, and full-data fidelity sequence. The lower fidelities allocate search effort; only a completed full-data stage can be declared empirically converged.

## fig06_convergence_gates.png

Figure 6. Advancement requires all convergence gates simultaneously: minimum trials, the full eligible data, the family-specific seed count, a flat final search tail, and a flat capacity tail. A failed gate remains visible rather than being converted into a poor score.

## fig07_primary_cell_budget.png

Figure 7. The roadmap bounds the scientific comparison at 68 primary target-architecture cells: 28 global cells, at most 30 additional spatial cells for five advancing families per target, and at most 10 target-representation cells. Folds, seeds, and trials are repetitions.

## fig08_information_policy.png

Figure 8. Main-ranking information is deliberately restricted. All families receive the same eligible rows within a target and deployment-valid nested SSS for fCO2; external target products, external pretraining, one-time independent labels, and 2026 labels are excluded.
