package coupledMCMC;

import beast.base.core.Description;
import beast.base.core.Input;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

@Description("Path sampling with coupled chains run in sequential batches. The steps of a batch run at the same time "
        + "as coupled chains; the chains of the next batch start from final states of the previous batch "
        + "instead of from the initial state (see warmStart), so only the first batch needs a long burn-in. "
        + "Steps are ordered from the posterior (beta = 1) to the prior (beta = 0). "
        + "The inherited preBurnin input is the number of samples the first batch runs before it starts logging.")
public class BatchedCoupledPathSampler extends CoupledPathSampler {
    public Input<Integer> batchSizeInput = new Input<>("batchSize", "number of steps (chains) that run at the same time, "
            + "for example the number of CPU cores. Default: all steps at once, which is a plain CoupledPathSampler");

    public enum WarmStart {
        /** every chain of a batch starts from the final state of the previous batch's hottest chain */
        hottest,
        /** chain i of a batch starts from the final state of chain i (the i-th coldest) of the previous batch */
        sameSlot
    }

    public Input<WarmStart> warmStartInput = new Input<>("warmStart", "which final states of the previous batch the "
            + "chains of the next batch start from: the hottest chain for all of them (hottest), or the chain with the "
            + "same position in the batch, coldest with coldest (sameSlot), which keeps the chains different from each other",
            WarmStart.hottest, WarmStart.values());

    private enum Start { INITIAL, OWN_FILES, PREVIOUS_BATCH }

    // what the hooks read while a batch runs
    private int batchOffset;
    private int batchIndex;
    private long batchLength;
    private boolean batchLogging;
    private Start batchStart = Start.INITIAL;

    @Override
    public void initAndValidate() {
        if (batchSizeInput.get() != null && batchSizeInput.get() < 1) {
            throw new IllegalArgumentException("batchSize must be at least 1");
        }
        if (burnInInput.get() < 0) {
            throw new IllegalArgumentException("preBurnin must not be negative");
        }
        super.initAndValidate();
    }

    /** the batches as {first step, number of steps}; the last one takes whatever is left */
    static int[][] batchRanges(int steps, int batchSize) {
        List<int[]> ranges = new ArrayList<>();
        for (int offset = 0; offset < steps; offset += batchSize) {
            ranges.add(new int[]{offset, Math.min(batchSize, steps - offset)});
        }
        return ranges.toArray(new int[0][]);
    }

    @Override
    protected int stepOf(int chainNr) {
        return batchOffset + chainNr;
    }

    @Override
    protected long chainLengthOfRun() {
        return batchLength;
    }

    @Override
    protected boolean restoreChainFromFile(int i) {
        return batchStart != Start.INITIAL;
    }

    @Override
    protected void configureChain(HeatedChain chain, int chainNr) {
        // the model's own loggers (the trace of the cold chain) belong to the first batch's real run only
        if (batchIndex > 0 || !batchLogging) {
            suppressAll(chain);
        }
        super.configureChain(chain, chainNr);
        if (!batchLogging) {
            suppressAll(chain);
        }
    }

    private static void suppressAll(HeatedChain chain) {
        for (CoupledLogger logger : chain.coupledLoggersInput.get()) {
            logger.setSuppressLogging(true);
        }
    }

    // first thing run() does, so a resume is refused before anything else looks at the old run
    @Override
    void checkOrWriteBetaSchedule() throws IOException {
        if (restoreFromFile) {
            throw new UnsupportedOperationException("BatchedCoupledPathSampler cannot resume a run: "
                    + "the state files of earlier batches are overwritten by later ones");
        }
        super.checkOrWriteBetaSchedule();
    }

    @Override
    protected void runAllSteps() throws IOException {
        int steps = betas.length;
        int[][] batches = batchRanges(steps, batchSizeInput.get() == null ? steps : batchSizeInput.get());
        long length = chainLengthInput.get();
        WarmStart mode = warmStartInput.get();
        int previousSize = 0;

        for (int b = 0; b < batches.length; b++) {
            batchIndex = b;
            batchOffset = batches[b][0];
            int n = batches[b][1];
            System.out.println("\nBatch " + (b + 1) + "/" + batches.length + ": steps " + batchOffset + "-" + (batchOffset + n - 1));

            if (b == 0 && burnInInput.get() > 0) {
                // nothing is logged; every chain leaves its state file behind, which the real run restores
                System.out.println("pre-burn-in: " + burnInInput.get() + " samples");
                runBatch(n, burnInInput.get(), false, Start.INITIAL);
                runBatch(n, length, true, Start.OWN_FILES);
            } else if (b == 0) {
                runBatch(n, length, true, Start.INITIAL);
            } else {
                // every chain restores the state, and the operator tuning, of the chain it continues
                for (int i = 0; i < n; i++) {
                    int source = mode == WarmStart.hottest ? previousSize - 1 : Math.min(i, previousSize - 1);
                    Files.copy(warmStartFile(source), Path.of(chainStateFileName(i)), StandardCopyOption.REPLACE_EXISTING);
                }
                runBatch(n, length, true, Start.PREVIOUS_BATCH);
            }

            if (b < batches.length - 1) {
                // slot n-1 is the hottest step; its file holds the final state of whichever chain ended there
                for (int slot = mode == WarmStart.hottest ? n - 1 : 0; slot < n; slot++) {
                    Files.copy(Path.of(chainStateFileName(slot)), warmStartFile(slot), StandardCopyOption.REPLACE_EXISTING);
                }
                previousSize = n;
            }
        }
        for (int slot = 0; slot < batches[0][1]; slot++) {
            Files.deleteIfExists(warmStartFile(slot));
        }
    }

    // the final state of a slot of the previous batch, kept because the next batch overwrites the slots' own files
    private Path warmStartFile(int slot) {
        return Path.of(stateFileName + ".warmstart" + slot);
    }

    private void runBatch(int n, long length, boolean logging, Start start) throws IOException {
        batchLength = length;
        batchLogging = logging;
        batchStart = start;
        chains = new HeatedChain[n];
        initRun();
        runChains();
        closeLikelihoodLoggers();
    }
}
