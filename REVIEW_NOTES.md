# Submission review notes

The vLLM `pr-checklist` skill was applied. This is a prepared contribution for
human review and submission, not an assertion that the human submitter has
already reviewed every line or personally rerun the tests.

## Design

The change is isolated to optional sampling dispatch and host-side top-k
metadata in MRV2. It does not change scheduling, KV-cache ownership, model
forward execution or rejection-sampling kernels. Shared APIs keep `k_max=None`
as the compatibility default. Capability and route checks are cached. No
steady-state GPU tensor is read back to choose the sampling backend.

Workspaces are cached per device by power-of-two batch capacity, and callers
get row views. Total memory stays within about 4x the largest batch
(12 KB per row). The buffers are never freed, so captured CUDA graphs keep
valid views. Mixed batches containing a row without top-k fall back as a whole.
Direct sampling and filtering are separate dispatches, because the extra
softmax and mask passes make the filtering route unattractive at larger row
counts. The 64-row cutoff is measured rather than a claim of universal optimality.

## Compatibility and validation

FlashInfer versions without Cake remain on the existing paths. CPU, non-CUDA,
unsupported Cake routes, missing host top-k metadata, and unsupported top-k
values also fall back. Cake is opt-in with
`VLLM_USE_FLASHINFER_CAKE_SAMPLER=1`; the e2e A/B did not resolve a serving
gain, so the default is unchanged. Activation also requires a FlashInfer
version that exposes the API; the dependency upgrade is separate work.

The test suite covers eligibility, fallback, retained support, tie behavior,
RNG isolation for the mask operation, and sampling probabilities. Real serving
probes verify that enabled and disabled arms take distinct paths. Probe wrappers
are removed before timing. The measured server enables the MRV2 GPU sync checker;
the only allowed tensor-to-host capture occurs in the explicit untimed probe.

Real-model probability checks and serving acceptance are covered. No downstream
task-quality or bitwise-token-equivalence claim is made. Exact-k tie handling
and softmax rounding can change the retained support versus the old reference.

## Duplicate work

The Cake tracker is #59725. Checks of its discussion and open PRs referencing
it found sparse-indexer and attention work, not this top-k/top-p integration.
The open min-p optimization #58907 addresses a different operation. #60948
updates FlashInfer and is complementary; this branch does not duplicate that
dependency bump.

Older #48927 and #48928 are research context for probability accuracy and AIR
sampling respectively. They are not required dependencies and no merge request
for those older experiments is implied by this contribution.

## Human submission

Repository `AGENTS.md` requires the submitting human to understand and defend
the change, review every changed line, and run the relevant tests. The public
source branch, filled PR description, measurements and commands are prepared
for that review. The description explicitly discloses Claude Code and OpenAI
Codex assistance.
