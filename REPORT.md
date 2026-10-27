# Report & Presentation Guide

## 1. Abstract (draft, edit with your final numbers)
We present a web-based system that classifies satellite imagery into ten land-use classes using a
fine-tuned ResNet18, explains predictions with Grad-CAM, detects changes between two acquisitions,
and uses the results to plan an imaging sequence under time constraints. The classifier reaches
96.07% test accuracy on EuroSAT. A greedy value-per-minute planner is compared with a random
baseline. Geography is synthetic unless supplied by the user; the pipeline accepts real inputs.

## 2. Report outline
1. **Introduction**: why automated land-use analysis and tasking matter (disaster monitoring, deforestation, urban growth).
2. **Problem statement and objectives**
3. **Dataset**: EuroSAT (27,000 Sentinel-2 RGB tiles, 64x64, 10 classes); 70/15/15 split, seed 42.
4. **Methodology**
   - 4.1 Transfer learning: ImageNet ResNet18, new 10-class head, Adam lr 1e-3, 5 epochs, tiles resized to 128.
   - 4.2 Explainability: Grad-CAM on the last conv block.
   - 4.3 Change detection: (a) per-band standardised pixel difference, smoothed, thresholded; (b) 4x4 grid classification, flag cells whose class changed.
   - 4.4 Mission planner: value = class priority x confidence x (1 - cloud); cost = imaging time + re-pointing distance/speed; greedy picks best value per minute until the time budget ends; baseline = random feasible choice averaged over 100 runs.
5. **Results**: test accuracy, per-class precision/recall (`metrics.json`), confusion matrix, Grad-CAM examples (successes and failures), change-detection example with known ground truth, greedy vs random (+X%) and value vs budget curve. **Fill in your own measured numbers.**
6. **Limitations**
   - Simulated coordinates and a heuristic cloud estimate.
   - EuroSAT tiles are small and clean; full scenes, haze, and other sensors will reduce accuracy.
   - Greedy is not optimal; no orbital mechanics or sensor/downlink constraints.
   - Pixel-difference change detection is sensitive to lighting and season; the grid method is coarse.
7. **Future work**: real Sentinel-2 scenes and cloud masks (SCL band), skyfield pass prediction, OR-Tools optimal scheduling vs greedy, semantic segmentation (DeepGlobe/LoveDA), Siamese change-detection networks (e.g. OSCD dataset).
8. **Conclusion**; 9. **References** (EuroSAT: Helber et al., 2019; ResNet: He et al., 2016; Grad-CAM: Selvaraju et al., 2017).

## 3. Slides (about 10)
Title and team / Problem / Dataset / System architecture (diagram) / Model and accuracy / Confusion matrix / Grad-CAM (include one failure) / Change detection / Mission planner and greedy vs random / Live demo / Limitations and future work.

## 4. Team split
Model and evaluation: `train.py`, metrics. Change detection and preprocessing: `core.py` change functions. Planner: `core.plan`. UI, report, slides: `app.py`, this document.

## 5. Questions faculty may ask
- *Why 96% and not 99%?* Highway/River and Pasture/HerbaceousVegetation look alike at 64 px; show the confusion matrix.
- *Is the test set clean?* Yes if you trained with `train.py` (seeded split; checkpoint chosen on validation, test used once at the end).
- *Is the planner optimal?* No, greedy heuristic; future work compares with OR-Tools.
- *Is the location real?* Only if supplied; otherwise simulated, and labelled in the table.
- *Why Grad-CAM?* To check the model looks at the right region (e.g. water for River), not background artefacts.
