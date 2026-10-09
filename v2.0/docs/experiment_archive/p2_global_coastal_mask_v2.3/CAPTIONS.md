# Figure captions

## fig01_candidate_gt0km2.png

Figure 1. Observation-independent 400-km candidate using every GSHHG island as land; this maximizes small-island halos and supplies the inclusive sensitivity endpoint.

## fig02_candidate_gt1400km2.png

Figure 2. Frozen ReCAD product geometry: ocean nodes within 400 km of GSHHG land at least 1,400 km², evaluated on the native 1/8-degree grid.

## fig03_candidate_gt4748km2.png

Figure 3. Restrictive 4,748-km² island-threshold candidate, included as the historical IBTrACS large-island sensitivity endpoint rather than the selected geometry.

## fig04_candidate_difference.png

Figure 4. Cells removed as the minimum island area increases; class 1 is present only in the all-island candidate and class 2 is retained at 1,400 km² but removed at 4,748 km².

## fig05_candidate_area_nodes.png

Figure 5. Total spherical area and native-grid node count for the three candidate product geometries, quantifying the consequence of the island-size choice.

## fig06_latitude_coverage.png

Figure 6. Spherical product area by ten-degree latitude band for each candidate, showing where the island threshold changes global coverage.

## fig07_observation_retention.png

Figure 7. Fraction of the frozen SOCATv2026 cache falling inside each candidate; observations are used only after geometry construction as a sensitivity audit and never define the mask.

## fig08_distance_bins.png

Figure 8. Area of the selected 1,400-km² product geometry by distance from significant land; the complete release envelope ends at 400 km.

## fig09_bathymetric_classes.png

Figure 9. GEBCO-2025 subdivision of the selected product into a 0-200 m shelf core, waters deeper than 200 m, and any bathymetrically unclassified cells.

## fig10_bathymetric_area.png

Figure 10. Spherical area of shelf, slope/deep, and unclassified cells within the selected 400-km product geometry.

## fig11_independent_distance_audit.png

Figure 11. Deterministic 2,000-point comparison between the GSHHG distance lookup and an independent haversine nearest-coast calculation from Natural Earth 10m vertices; disagreement reflects both source geometry and raster resolution.
