(venv) yutokuroki@Yutos-MacBook-Air TCN implementation % python real_gait_tcn/cross_validation.py \
  --num-users 0 \
  --splits 5 \
  --epochs 5 \
  --batch-size 32 \
  --sequence-stride 100 \
  --latency-batches 20 \
  --latency-iterations 10 \
  --plot-samples 500 \
  --plot-width 24 \
  --plot-height 4 \
  --plot-dir results/attention_comparison/plots/large_test \
  --results-file results/attention_comparison/json/cv_large_test.json
Testing with all 34 users

======================================================================
USER-STRATIFIED 5-FOLD CROSS-VALIDATION
======================================================================
Total participants: 34
Sequence length: 100
Sequence stride: 100
Input features: 14
Target: hip_moment_l

Device: mps
Latency benchmark: enabled (20 batches, 10 iterations/batch)

Prediction plots: enabled (500 samples/fold, 24.0x4.0 in, dir=results/attention_comparison/plots/large_test)

======================================================================
FOLD 1/5
======================================================================
Training participants: 27 users
Validation participants: 7 users

  Loading train participants: ['AB01', 'AB02', 'AB03', 'AB04', 'AB05', 'AB06', 'AB07', 'AB08', 'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB17', 'AB18', 'AB19', 'AB21', 'AB23', 'AB24', 'AB26', 'AB29', 'AB30', 'AB31', 'AB32', 'AB33', 'AB34']
    OK AB01: 703 samples
    OK AB02: 660 samples
    OK AB03: 668 samples
    OK AB04: 646 samples
    OK AB05: 434 samples
    OK AB06: 692 samples
    OK AB07: 672 samples
    OK AB08: 689 samples
    OK AB10: 829 samples
    OK AB11: 753 samples
    OK AB12: 704 samples
    OK AB13: 748 samples
    OK AB14: 684 samples
    OK AB15: 1357 samples
    OK AB17: 1381 samples
    OK AB18: 1329 samples
    OK AB19: 1457 samples
    OK AB21: 1648 samples
    OK AB23: 1643 samples
    OK AB24: 1670 samples
    OK AB26: 1650 samples
    OK AB29: 1593 samples
    OK AB30: 1982 samples
    OK AB31: 1765 samples
    OK AB32: 1853 samples
    OK AB33: 1921 samples
    OK AB34: 2120 samples
  Loading val participants: ['AB09', 'AB16', 'AB20', 'AB22', 'AB25', 'AB27', 'AB28']
    OK AB09: 741 samples
    OK AB16: 1564 samples
    OK AB20: 1652 samples
    OK AB22: 1651 samples
    OK AB25: 1551 samples
    OK AB27: 1615 samples
    OK AB28: 1459 samples

Training samples: 32251, Val samples: 10233

  Training TCN WITHOUT Attention...
      [WITHOUT] Epoch 01/5 | Train: 0.040594 | Val: 0.033100
      [WITHOUT] Epoch 02/5 | Train: 0.029631 | Val: 0.031122
      [WITHOUT] Epoch 03/5 | Train: 0.027101 | Val: 0.029586
      [WITHOUT] Epoch 04/5 | Train: 0.025035 | Val: 0.027554
      [WITHOUT] Epoch 05/5 | Train: 0.024273 | Val: 0.028916
  Best Val Loss (WITHOUT): 0.027554
  Latency (WITHOUT, mps): 0.019018 ms/sample, 52580.6 samples/sec

  Training TCN WITH Attention...
      [WITH] Epoch 01/5 | Train: 0.036090 | Val: 0.030805
      [WITH] Epoch 02/5 | Train: 0.024116 | Val: 0.026393
      [WITH] Epoch 03/5 | Train: 0.020613 | Val: 0.025646
      [WITH] Epoch 04/5 | Train: 0.018157 | Val: 0.027100
      [WITH] Epoch 05/5 | Train: 0.017058 | Val: 0.025066
  Best Val Loss (WITH): 0.025066
  Latency (WITH, mps): 0.028633 ms/sample, 34925.2 samples/sec
  Prediction plot saved to: results/attention_comparison/plots/large_test/fold_01_moment_predictions.png

  Improvement: +9.03%
  Attention latency overhead: +50.55%
======================================================================
FOLD 2/5
======================================================================
Training participants: 27 users
Validation participants: 7 users

  Loading train participants: ['AB02', 'AB03', 'AB04', 'AB06', 'AB07', 'AB08', 'AB09', 'AB11', 'AB12', 'AB14', 'AB15', 'AB16', 'AB19', 'AB20', 'AB21', 'AB22', 'AB23', 'AB24', 'AB25', 'AB26', 'AB27', 'AB28', 'AB29', 'AB30', 'AB31', 'AB32', 'AB34']
    OK AB02: 660 samples
    OK AB03: 668 samples
    OK AB04: 646 samples
    OK AB06: 692 samples
    OK AB07: 672 samples
    OK AB08: 689 samples
    OK AB09: 741 samples
    OK AB11: 753 samples
    OK AB12: 704 samples
    OK AB14: 684 samples
    OK AB15: 1357 samples
    OK AB16: 1564 samples
    OK AB19: 1457 samples
    OK AB20: 1652 samples
    OK AB21: 1648 samples
    OK AB22: 1651 samples
    OK AB23: 1643 samples
    OK AB24: 1670 samples
    OK AB25: 1551 samples
    OK AB26: 1650 samples
    OK AB27: 1615 samples
    OK AB28: 1459 samples
    OK AB29: 1593 samples
    OK AB30: 1982 samples
    OK AB31: 1765 samples
    OK AB32: 1853 samples
    OK AB34: 2120 samples
  Loading val participants: ['AB01', 'AB05', 'AB10', 'AB13', 'AB17', 'AB18', 'AB33']
    OK AB01: 703 samples
    OK AB05: 434 samples
    OK AB10: 829 samples
    OK AB13: 748 samples
    OK AB17: 1381 samples
    OK AB18: 1329 samples
    OK AB33: 1921 samples

Training samples: 35139, Val samples: 7345

  Training TCN WITHOUT Attention...
      [WITHOUT] Epoch 01/5 | Train: 0.038646 | Val: 0.031291
      [WITHOUT] Epoch 02/5 | Train: 0.028580 | Val: 0.029971
      [WITHOUT] Epoch 03/5 | Train: 0.026185 | Val: 0.027519
      [WITHOUT] Epoch 04/5 | Train: 0.024792 | Val: 0.027964
      [WITHOUT] Epoch 05/5 | Train: 0.023648 | Val: 0.027113
  Best Val Loss (WITHOUT): 0.027113
  Latency (WITHOUT, mps): 0.018358 ms/sample, 54472.0 samples/sec

  Training TCN WITH Attention...
      [WITH] Epoch 01/5 | Train: 0.035236 | Val: 0.028782
      [WITH] Epoch 02/5 | Train: 0.023993 | Val: 0.026585
      [WITH] Epoch 03/5 | Train: 0.020975 | Val: 0.026417
      [WITH] Epoch 04/5 | Train: 0.018385 | Val: 0.022828
      [WITH] Epoch 05/5 | Train: 0.017142 | Val: 0.022764
  Best Val Loss (WITH): 0.022764
  Latency (WITH, mps): 0.040770 ms/sample, 24527.6 samples/sec
  Prediction plot saved to: results/attention_comparison/plots/large_test/fold_02_moment_predictions.png

  Improvement: +16.04%
  Attention latency overhead: +122.08%
======================================================================
FOLD 3/5
======================================================================
Training participants: 27 users
Validation participants: 7 users

  Loading train participants: ['AB01', 'AB05', 'AB07', 'AB08', 'AB09', 'AB10', 'AB11', 'AB13', 'AB15', 'AB16', 'AB17', 'AB18', 'AB19', 'AB20', 'AB21', 'AB22', 'AB23', 'AB24', 'AB25', 'AB26', 'AB27', 'AB28', 'AB29', 'AB30', 'AB32', 'AB33', 'AB34']
    OK AB01: 703 samples
    OK AB05: 434 samples
    OK AB07: 672 samples
    OK AB08: 689 samples
    OK AB09: 741 samples
    OK AB10: 829 samples
    OK AB11: 753 samples
    OK AB13: 748 samples
    OK AB15: 1357 samples
    OK AB16: 1564 samples
    OK AB17: 1381 samples
    OK AB18: 1329 samples
    OK AB19: 1457 samples
    OK AB20: 1652 samples
    OK AB21: 1648 samples
    OK AB22: 1651 samples
    OK AB23: 1643 samples
    OK AB24: 1670 samples
    OK AB25: 1551 samples
    OK AB26: 1650 samples
    OK AB27: 1615 samples
    OK AB28: 1459 samples
    OK AB29: 1593 samples
    OK AB30: 1982 samples
    OK AB32: 1853 samples
    OK AB33: 1921 samples
    OK AB34: 2120 samples
  Loading val participants: ['AB02', 'AB03', 'AB04', 'AB06', 'AB12', 'AB14', 'AB31']
    OK AB02: 660 samples
    OK AB03: 668 samples
    OK AB04: 646 samples
    OK AB06: 692 samples
    OK AB12: 704 samples
    OK AB14: 684 samples
    OK AB31: 1765 samples

Training samples: 36665, Val samples: 5819

  Training TCN WITHOUT Attention...
      [WITHOUT] Epoch 01/5 | Train: 0.036586 | Val: 0.047393
      [WITHOUT] Epoch 02/5 | Train: 0.026410 | Val: 0.046689
      [WITHOUT] Epoch 03/5 | Train: 0.024606 | Val: 0.045152
      [WITHOUT] Epoch 04/5 | Train: 0.023128 | Val: 0.047651
      [WITHOUT] Epoch 05/5 | Train: 0.022130 | Val: 0.046706
  Best Val Loss (WITHOUT): 0.045152
  Latency (WITHOUT, mps): 0.018529 ms/sample, 53968.4 samples/sec

  Training TCN WITH Attention...
      [WITH] Epoch 01/5 | Train: 0.030069 | Val: 0.046236
      [WITH] Epoch 02/5 | Train: 0.020762 | Val: 0.046797
      [WITH] Epoch 03/5 | Train: 0.018079 | Val: 0.047170
      [WITH] Epoch 04/5 | Train: 0.016293 | Val: 0.042806
      [WITH] Epoch 05/5 | Train: 0.014717 | Val: 0.047124
  Best Val Loss (WITH): 0.042806
  Latency (WITH, mps): 0.029454 ms/sample, 33951.4 samples/sec
  Prediction plot saved to: results/attention_comparison/plots/large_test/fold_03_moment_predictions.png

  Improvement: +5.20%
  Attention latency overhead: +58.96%
======================================================================
FOLD 4/5
======================================================================
Training participants: 27 users
Validation participants: 7 users

  Loading train participants: ['AB01', 'AB02', 'AB03', 'AB04', 'AB05', 'AB06', 'AB08', 'AB09', 'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17', 'AB18', 'AB20', 'AB21', 'AB22', 'AB25', 'AB27', 'AB28', 'AB29', 'AB31', 'AB33', 'AB34']
    OK AB01: 703 samples
    OK AB02: 660 samples
    OK AB03: 668 samples
    OK AB04: 646 samples
    OK AB05: 434 samples
    OK AB06: 692 samples
    OK AB08: 689 samples
    OK AB09: 741 samples
    OK AB10: 829 samples
    OK AB11: 753 samples
    OK AB12: 704 samples
    OK AB13: 748 samples
    OK AB14: 684 samples
    OK AB15: 1357 samples
    OK AB16: 1564 samples
    OK AB17: 1381 samples
    OK AB18: 1329 samples
    OK AB20: 1652 samples
    OK AB21: 1648 samples
    OK AB22: 1651 samples
    OK AB25: 1551 samples
    OK AB27: 1615 samples
    OK AB28: 1459 samples
    OK AB29: 1593 samples
    OK AB31: 1765 samples
    OK AB33: 1921 samples
    OK AB34: 2120 samples
  Loading val participants: ['AB07', 'AB19', 'AB23', 'AB24', 'AB26', 'AB30', 'AB32']
    OK AB07: 672 samples
    OK AB19: 1457 samples
    OK AB23: 1643 samples
    OK AB24: 1670 samples
    OK AB26: 1650 samples
    OK AB30: 1982 samples
    OK AB32: 1853 samples

Training samples: 31557, Val samples: 10927

  Training TCN WITHOUT Attention...
      [WITHOUT] Epoch 01/5 | Train: 0.042702 | Val: 0.023958
      [WITHOUT] Epoch 02/5 | Train: 0.031946 | Val: 0.022584
      [WITHOUT] Epoch 03/5 | Train: 0.028896 | Val: 0.021039
      [WITHOUT] Epoch 04/5 | Train: 0.026970 | Val: 0.020608
      [WITHOUT] Epoch 05/5 | Train: 0.026189 | Val: 0.019413
  Best Val Loss (WITHOUT): 0.019413
  Latency (WITHOUT, mps): 0.018217 ms/sample, 54894.3 samples/sec

  Training TCN WITH Attention...
      [WITH] Epoch 01/5 | Train: 0.037350 | Val: 0.022201
      [WITH] Epoch 02/5 | Train: 0.024344 | Val: 0.019795
      [WITH] Epoch 03/5 | Train: 0.021200 | Val: 0.018083
      [WITH] Epoch 04/5 | Train: 0.018880 | Val: 0.015794
      [WITH] Epoch 05/5 | Train: 0.017518 | Val: 0.016342
  Best Val Loss (WITH): 0.015794
  Latency (WITH, mps): 0.033399 ms/sample, 29941.4 samples/sec
  Prediction plot saved to: results/attention_comparison/plots/large_test/fold_04_moment_predictions.png

  Improvement: +18.64%
  Attention latency overhead: +83.34%
======================================================================
FOLD 5/5
======================================================================
Training participants: 28 users
Validation participants: 6 users

  Loading train participants: ['AB01', 'AB02', 'AB03', 'AB04', 'AB05', 'AB06', 'AB07', 'AB09', 'AB10', 'AB12', 'AB13', 'AB14', 'AB16', 'AB17', 'AB18', 'AB19', 'AB20', 'AB22', 'AB23', 'AB24', 'AB25', 'AB26', 'AB27', 'AB28', 'AB30', 'AB31', 'AB32', 'AB33']
    OK AB01: 703 samples
    OK AB02: 660 samples
    OK AB03: 668 samples
    OK AB04: 646 samples
    OK AB05: 434 samples
    OK AB06: 692 samples
    OK AB07: 672 samples
    OK AB09: 741 samples
    OK AB10: 829 samples
    OK AB12: 704 samples
    OK AB13: 748 samples
    OK AB14: 684 samples
    OK AB16: 1564 samples
    OK AB17: 1381 samples
    OK AB18: 1329 samples
    OK AB19: 1457 samples
    OK AB20: 1652 samples
    OK AB22: 1651 samples
    OK AB23: 1643 samples
    OK AB24: 1670 samples
    OK AB25: 1551 samples
    OK AB26: 1650 samples
    OK AB27: 1615 samples
    OK AB28: 1459 samples
    OK AB30: 1982 samples
    OK AB31: 1765 samples
    OK AB32: 1853 samples
    OK AB33: 1921 samples
  Loading val participants: ['AB08', 'AB11', 'AB15', 'AB21', 'AB29', 'AB34']
    OK AB08: 689 samples
    OK AB11: 753 samples
    OK AB15: 1357 samples
    OK AB21: 1648 samples
    OK AB29: 1593 samples
    OK AB34: 2120 samples

Training samples: 34324, Val samples: 8160

  Training TCN WITHOUT Attention...
      [WITHOUT] Epoch 01/5 | Train: 0.041829 | Val: 0.030747
      [WITHOUT] Epoch 02/5 | Train: 0.031314 | Val: 0.028487
      [WITHOUT] Epoch 03/5 | Train: 0.028255 | Val: 0.025467
      [WITHOUT] Epoch 04/5 | Train: 0.026447 | Val: 0.024957
      [WITHOUT] Epoch 05/5 | Train: 0.025471 | Val: 0.025756
  Best Val Loss (WITHOUT): 0.024957
  Latency (WITHOUT, mps): 0.017962 ms/sample, 55674.6 samples/sec

  Training TCN WITH Attention...
      [WITH] Epoch 01/5 | Train: 0.034692 | Val: 0.028709
      [WITH] Epoch 02/5 | Train: 0.023395 | Val: 0.023374
      [WITH] Epoch 03/5 | Train: 0.020201 | Val: 0.023881
      [WITH] Epoch 04/5 | Train: 0.018581 | Val: 0.025846
      [WITH] Epoch 05/5 | Train: 0.016701 | Val: 0.019995
  Best Val Loss (WITH): 0.019995
  Latency (WITH, mps): 0.029568 ms/sample, 33820.5 samples/sec
  Prediction plot saved to: results/attention_comparison/plots/large_test/fold_05_moment_predictions.png

  Improvement: +19.88%
  Attention latency overhead: +64.62%

======================================================================
CROSS-VALIDATION SUMMARY
======================================================================

Fold     WITHOUT (Val Loss)   WITH (Val Loss)      Improvement    
----------------------------------------------------------------------
1        0.027554             0.025066                      +9.03%
2        0.027113             0.022764                     +16.04%
3        0.045152             0.042806                      +5.20%
4        0.019413             0.015794                     +18.64%
5        0.024957             0.019995                     +19.88%
----------------------------------------------------------------------
AVG      0.028838             0.025285                     +13.76%
STD      0.008656             0.009289                       5.70%

======================================================================
ANALYSIS
======================================================================

Attention Mechanism Impact on User Generalization:
  Average validation loss WITHOUT attention: 0.028838 ± 0.008656
  Average validation loss WITH attention:    0.025285 ± 0.009289
  Average improvement:                       +13.76% ± 5.70%

Inference Latency Impact (mps):
  Average latency WITHOUT attention:         0.018417 ms/sample
  Average latency WITH attention:            0.032365 ms/sample
  Average attention latency overhead:        +75.91%
  Average throughput WITHOUT attention:      54318.0 samples/sec
  Average throughput WITH attention:         31433.2 samples/sec
  Average throughput reduction:              +42.07%

Attention IMPROVES generalization to unseen users
  The model with attention adapts better to different user biomechanics

Results saved to: results/attention_comparison/json/cv_large_test.json
(venv) yutokuroki@Yutos-MacBook-Air TCN implementation % 