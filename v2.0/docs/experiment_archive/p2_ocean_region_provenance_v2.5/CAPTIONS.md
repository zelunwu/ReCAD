# Figure captions

## fig01_corrected_region_map.png

Figure 1. Polygon-based Global Oceans and Seas v1 provenance for every ReCAD v2.4 coastal node. The ten source regions are mutually auditable reporting strata and do not directly determine reconstruction quality or release status.

## fig02_legacy_basin_map.png

Figure 2. Legacy v2.4 basin labels produced by latitude and longitude thresholds. The 145 degree east boundary cuts through the western North Pacific and assigns part of Japan and adjacent seas to the Indian label.

## fig03_changed_nodes_map.png

Figure 3. Nodes whose reporting provenance changes when the coarse threshold rule is replaced by GOaS polygons. Purple denotes a changed macro label or a newly explicit Baltic or archipelagic region; gray denotes agreement.

## fig04_japan_corrected_map.png

Figure 4. Japan-area product nodes assigned from the GOaS North Pacific polygon. Both the Pacific-facing and Sea of Japan margins receive North Pacific provenance, removing the legacy Indian-label bypass.

## fig05_confusion_matrix.png

Figure 5. Node-count cross-tabulation between legacy coarse basins and new reporting macro basins. Cell labels are raw counts and color is logarithmic, exposing substantial off-diagonal assignments hidden by the old longitude boundaries.

## fig06_region_node_counts.png

Figure 6. Number of ReCAD coastal nodes in each GOaS source region. These counts describe geographic product support only and are not weights, quality grades, or target-observation counts.

## fig07_unresolved_nodes.png

Figure 7. Product nodes not covered by the frozen GOaS polygon geometry. They remain explicitly unresolved rather than receiving a fallback basin from longitude, and must be investigated before provenance is written into a release product.

## fig08_shared_boundary_nodes.png

Figure 8. Product nodes covered by more than one GOaS polygon because they lie exactly on a shared boundary. The stable lower region ID is retained for serialization while the boundary flag preserves the ambiguity for audits.
