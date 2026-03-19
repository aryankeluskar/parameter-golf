# Parameter Golf Challenge - Implementation Plan

## Objective
Beat current SOTA (val_bpb=1.1748) by >=0.005 on 8xH100 in 10 minutes, <16MB artifact.

## Current SOTA Analysis (1.1748)
The leading submission combines:
1. Sliding window eval (stride=64) - free ~0.032 BPB from better eval context
2. FP16 tied embeddings - avoids double int8 damage on embedding
3. 10 layers (up from 9) - Muon WD compresses enough to fit extra layer
4. Muon weight decay (0.02) - better generalization + quantization robustness
5. Overtone spectral embedding init - SVD power-law spectrum
6. Phase-transition residual mixing - sigmoid-scheduled resid_mix
7. U-Net skip connections between encoder/decoder halves
8. Architecture: 10L/512dim/8heads/4KV/2x MLP/1024vocab/tied embeddings

## Strategy to Beat 1.1748 (Target: <=1.1698)

### High-Confidence Improvements (stack these)

#### 1. Longer Training Context (seq_len=2048 or 4096 during training)
- The seq4096 submission got 1.2014 with 9 layers and no sliding eval
- Combined with SOTA's other improvements (10L, sliding eval, fp16 embed, Muon WD), this should yield significant gains
- seq_len=2048 is safer (less ms/step overhead), 4096 is riskier but higher upside
- **Expected gain: 0.01-0.03 BPB**

#### 2. MLP Expansion 3x (from 2x)
- The Int6+MLP3x submission (PR #61, commit 555669e) achieved 1.1574
- 3x MLP hidden gives more capacity per parameter; may need to adjust layer count to fit 16MB
- **Expected gain: 0.005-0.015 BPB**

#### 3. Aggressive Warmdown (WARMDOWN_ITERS=20000)
- WarmdownQuantization submission showed this dramatically reduces post-quant penalty
- Entire training run in decay phase → tighter weight distributions → better int8 quantization
- **Expected gain: 0.005-0.010 BPB on post-quant score**

#### 4. Larger Vocab (2048 or 4096)
- Larger vocab = fewer tokens per byte = more efficient compression
- Needs careful BPB calculation verification
- Costs more embedding parameters (offset by tied embeddings + fp16 storage)
- **Speculative, needs testing**

#### 5. Hyperparameter Tuning
- Higher Muon momentum (0.99 vs 0.95) as in seq4096 submission
- Tuned learning rates for the specific architecture
- Gradient clipping (GRAD_CLIP_NORM=1.0 helped in WarmdownQuantization)

### Approach: Combine Best of All Submissions

Start from SOTA code, add:
1. `TRAIN_SEQ_LEN=2048` (or 4096)
2. `MLP_MULT=3` with layer count adjusted to fit 16MB
3. `WARMDOWN_ITERS=20000`
4. `MUON_MOMENTUM=0.99`, `MUON_MOMENTUM_WARMUP_STEPS=1500`
5. Lower LRs to reduce quant penalty
6. Keep: sliding eval, fp16 embed, overtone init, resid mix, Muon WD, skip connections

## Implementation Steps

### Phase 1: Setup RunPod (User Action Required)
1. Launch 1xH100 pod for iteration using the Parameter Golf template
2. SSH into pod, clone repo, download data
3. Run baseline to verify setup

### Phase 2: Create Submission Script
1. Fork SOTA's train_gpt.py as our starting point
2. Integrate seq_len=2048+ training from seq4096 submission
3. Add MLP_MULT=3, adjust NUM_LAYERS to fit 16MB budget
4. Set WARMDOWN_ITERS=20000
5. Tune optimizer hyperparameters

### Phase 3: Iterate on 1xH100
1. Quick smoke tests (2-3 min runs) to verify no crashes
2. Compare pre-quant val_bpb across configurations
3. Check compressed model size stays under 16MB
4. Key experiments to run:
   - A: SOTA + seq2048 only
   - B: SOTA + seq2048 + MLP3x (fewer layers if needed)
   - C: SOTA + seq2048 + warmdown20k
   - D: Best of above combined

### Phase 4: Full 8xH100 Validation Runs
1. Launch 8xH100 pod
2. Run 3 seeds of best config
3. Verify val_bpb < 1.1698 with p < 0.01
4. Verify artifact < 16MB
5. Verify total time < 10 min

### Phase 5: Prepare Submission
1. Create records folder: `records/track_10min_16mb/YYYY-MM-DD_OurSubmission/`
2. Include: train_gpt.py, README.md, submission.json, train.log (3 seeds)
3. Open PR

## Key Files
| File | Operation | Description |
|------|-----------|-------------|
| records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py | Read (base) | SOTA code to fork |
| records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py | Read (reference) | Seq4096 + Muon tuning |
| New: records/track_10min_16mb/YYYY-MM-DD_OurSubmission/train_gpt.py | Create | Our submission |

## Risks and Mitigation
| Risk | Mitigation |
|------|------------|
| MLP 3x + 10 layers exceeds 16MB | Reduce to 9 or 8 layers, or reduce dim to 480 |
| Longer seq_len too slow per step | Fall back to seq2048 instead of 4096 |
| Combined changes destabilize training | Test incrementally, one change at a time |
| Quant penalty wipes gains | Use warmdown20k + lower LRs + fp16 embed |
| RunPod 8xH100 availability | Try multiple regions, off-peak hours |

## What I Need From You
1. **RunPod account access** - Do you have a RunPod account set up? Do you have credits/budget?
2. **SSH key configured** on RunPod for pod access
3. **GitHub fork** of openai/parameter-golf for submitting the PR

## Parameter Budget Estimate
- Embedding: 1024 * 512 = 524K params (fp16 = 1MB)
- Per layer (MLP 3x): Q/K/V/O (4 * 512^2) + MLP (2 * 512 * 1536) = ~2.6M params
- 10 layers: ~26M params → int8 = ~26MB pre-compression → ~15MB zlib compressed
- 8 layers with MLP 3x: ~21M params → ~12MB compressed → fits easily
- 9 layers with MLP 3x: ~23M params → should be ~14MB compressed → tight but should fit

Best starting point: 9 layers, MLP 3x, dim 512, and verify size.
