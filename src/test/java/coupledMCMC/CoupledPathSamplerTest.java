package coupledMCMC;

import beast.base.inference.CompoundDistribution;
import beast.base.inference.Logger;
import beast.base.inference.State;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.distribution.Normal;
import beast.base.spec.inference.operator.RealRandomWalkOperator;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import org.apache.commons.statistics.distribution.BetaDistribution;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class CoupledPathSamplerTest {

    // y = 1 observed, y | mu ~ N(mu, 1), mu ~ N(0, 1)  =>  y ~ N(0, 2)
    static CoupledPathSampler normalNormalSampler(Path dir, int chains, long chainLength) {
        return normalNormalSampler(dir, dir.toString(), dir.resolve("trace.log").toString(), chains, chainLength);
    }

    static CoupledPathSampler normalNormalSampler(Path dir, String rootDir, String traceFile, int chains, long chainLength) {
        RealScalarParam<Real> mu = new RealScalarParam<>(0.0, Real.INSTANCE);
        mu.setID("mu");
        Normal prior = new Normal();
        prior.initByName("param", mu, "mean", new RealScalarParam<>(0.0, Real.INSTANCE),
                "sigma", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));
        prior.setID("prior");
        Normal likelihood = new Normal();
        likelihood.initByName("param", new RealScalarParam<>(1.0, Real.INSTANCE), "mean", mu,
                "sigma", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));
        likelihood.setID("likelihood");
        CompoundDistribution posterior = new CompoundDistribution();
        posterior.initByName("distribution", prior, "distribution", likelihood);
        posterior.setID("posterior");
        State state = new State();
        state.initByName("stateNode", mu);
        RealRandomWalkOperator op = new RealRandomWalkOperator();
        op.initByName("scalar", mu, "windowSize", 1.0, "weight", 1.0);
        Logger trace = new Logger();
        trace.initByName("fileName", traceFile, "log", mu, "logEvery", 1000);
        trace.setID("tracelog");

        CoupledPathSampler sampler = new CoupledPathSampler();
        sampler.initByName("chains", chains, "chainLength", chainLength, "resampleEvery", 100,
                "rootdir", rootDir, "alpha", 0.3, "burnInPercentage", 10,
                "distribution", posterior, "state", state, "operator", op, "logger", trace);
        sampler.setStateFile(dir.resolve("cps.state").toString(), false);
        return sampler;
    }

    static void runAndCleanUp(CoupledPathSampler sampler, int chains) throws Exception {
        Logger.FILE_MODE = Logger.LogFileMode.overwrite;
        try {
            sampler.run();
        } finally {
            // CoupledMCMC writes per-chain state files to the working directory, by file name only
            for (int k = 0; k < chains; k++) {
                Files.deleteIfExists(Path.of("cps." + k + "state"));
            }
        }
    }

    @Test
    void recoversAnalyticMarginalLikelihoodWithSinglePairSwaps(@TempDir Path dir) throws Exception {
        Randomizer.setSeed(127);
        CoupledPathSampler sampler = normalNormalSampler(dir, 12, 200_000);
        sampler.evenOddSwapsInput.setValue(false, sampler);
        runAndCleanUp(sampler, 12);
        assertEquals(2000, sampler.totalSwaps);
        double expected = -0.5 * Math.log(4 * Math.PI) - 0.25;
        assertEquals(expected, sampler.logMarginalLikelihood, 0.05);
        assertTrue(sampler.successfullSwaps > 0);
        for (HeatedChain chain : sampler.chains) {
            assertInstanceOf(HeatedChainLikelihoodOnly.class, chain);
        }
        for (int k = 0; k < 12; k++) {
            assertTrue(new File(sampler.stepDir(k), CoupledPathSampler.LIKELIHOOD_LOG_FILE).exists());
        }
    }

    @Test
    void evenOddRoundsAlternateNeighbourPairs() {
        assertArrayEquals(new int[][]{{0, 1}, {2, 3}}, EvenOddSwapRunner.neighbourPairs(5, 0));
        assertArrayEquals(new int[][]{{1, 2}, {3, 4}}, EvenOddSwapRunner.neighbourPairs(5, 1));
        assertArrayEquals(new int[][]{{0, 1}, {2, 3}}, EvenOddSwapRunner.neighbourPairs(4, 2));
        assertArrayEquals(new int[][]{{1, 2}}, EvenOddSwapRunner.neighbourPairs(4, 3));
        assertArrayEquals(new int[][]{}, EvenOddSwapRunner.neighbourPairs(2, 1));
    }

    @Test
    void evenOddSwapsAttemptEveryPairEachRound(@TempDir Path dir) throws Exception {
        Randomizer.setSeed(127);
        CoupledPathSampler sampler = normalNormalSampler(dir, 12, 200_000);
        runAndCleanUp(sampler, 12);
        // 1999 swap rounds (none after the last segment), alternating 6 even and 5 odd pairs
        assertEquals(1000 * 6 + 999 * 5, sampler.totalSwaps);
        assertTrue(sampler.successfullSwaps > 0);
        for (HeatedChain chain : sampler.chains) {
            assertEquals(sampler.betas[chain.getChainNr()], chain.getBeta());
        }
        double expected = -0.5 * Math.log(4 * Math.PI) - 0.25;
        assertEquals(expected, sampler.logMarginalLikelihood, 0.05);
    }

    @Test
    void failingChainAbortsRunInsteadOfReportingAnEstimate(@TempDir Path dir) {
        CoupledPathSampler sampler = normalNormalSampler(dir, 3, 2000);
        // Logger.log divides by logEvery, so every chain thread throws on its first log call
        sampler.likelihoodLogEveryInput.setValue(0, sampler);
        RuntimeException e = assertThrows(RuntimeException.class, () -> runAndCleanUp(sampler, 3));
        assertInstanceOf(ArithmeticException.class, e.getCause());
        assertTrue(Double.isNaN(sampler.logMarginalLikelihood));
    }

    @Test
    void resumeWithDifferentBetaScheduleIsRejected(@TempDir Path dir) throws Exception {
        runAndCleanUp(normalNormalSampler(dir, 3, 2000), 3);
        CoupledPathSampler resumed = normalNormalSampler(dir, 4, 2000);
        resumed.setStateFile(dir.resolve("cps.state").toString(), true);
        assertThrows(IllegalStateException.class, () -> runAndCleanUp(resumed, 4));
    }

    @Test
    void honoursBeastFileNamePrefix(@TempDir Path dir) throws Exception {
        // BEAST's -working / -prefix options set file.name.prefix, which Logger prepends to every log file
        System.setProperty("file.name.prefix", dir + File.separator);
        try {
            CoupledPathSampler sampler = normalNormalSampler(dir, "cps", "trace.log", 3, 2000);
            sampler.setStateFile("cps.state", false);
            runAndCleanUp(sampler, 3);
            assertTrue(Files.exists(dir.resolve("cps/step0/" + CoupledPathSampler.LIKELIHOOD_LOG_FILE)));
            assertFalse(Double.isNaN(sampler.logMarginalLikelihood));
        } finally {
            System.clearProperty("file.name.prefix");
        }
    }

    @Test
    void rejectsSettingsThatBreakPathSampling(@TempDir Path dir) {
        // initByName wraps the IllegalArgumentException
        assertThrows(RuntimeException.class, () -> normalNormalSampler(dir, 1, 1000));
        CoupledPathSampler s = new CoupledPathSampler();
        s.optimiseInput.setValue(true, s);
        assertThrows(IllegalArgumentException.class, s::initAndValidate);
        CoupledPathSampler s2 = new CoupledPathSampler();
        s2.heatLikelihoodOnlyInput.setValue(false, s2);
        assertThrows(IllegalArgumentException.class, s2::initAndValidate);
        CoupledPathSampler s3 = new CoupledPathSampler();
        s3.burnInPercentageInput.setValue(100, s3);
        assertThrows(IllegalArgumentException.class, s3::initAndValidate);
    }

    @Test
    void betaScheduleMatchesPathSampler() {
        double[] b = CoupledPathSampler.betaSchedule(5, 0.3);
        assertEquals(1.0, b[0]);
        assertEquals(0.0, b[4]);
        BetaDistribution d = BetaDistribution.of(0.3, 1.0);
        for (int k = 0; k < 5; k++) {
            assertEquals(d.inverseCumulativeProbability((4.0 - k) / 4), b[k], 1e-15);
        }
        assertArrayEquals(new double[]{1.0, 0.75, 0.5, 0.25, 0.0}, CoupledPathSampler.betaSchedule(5, 0.0), 1e-15);
    }

    @Test
    void steppingStoneOfConstantLikelihoodIsThatConstant() {
        double[] betas = {0.0, 0.1, 0.5, 1.0};
        double[][] logLs = {{-3, -3}, {-3, -3}, {-3, -3}, {-3, -3}};
        double sum = 0;
        for (double c : CoupledPathSampler.steppingStoneContributions(betas, logLs)) {
            sum += c;
        }
        assertEquals(-3.0, sum, 1e-12);
    }

    @Test
    void thermodynamicIntegrationIsTrapezoid() {
        double[] betas = {0.0, 0.5, 1.0};
        double[] means = {-10.0, -4.0, -2.0};
        assertEquals(0.5 * (-10 - 4) / 2 + 0.5 * (-4 - 2) / 2,
                CoupledPathSampler.thermodynamicIntegration(betas, means), 1e-12);
    }

    @Test
    void readTraceSkipsCommentsAndBurnIn(@TempDir Path dir) throws Exception {
        File f = dir.resolve("likelihood.log").toFile();
        Files.writeString(f.toPath(), "# comment\nSample\tlikelihood\t\n0\t-1.0\t\n10\t-2.0\t\n20\t-3.0\t\n30\t-4.0\t\n");
        assertEquals(List.of(-3.0, -4.0), CoupledPathSampler.readTrace(f, "likelihood", 50));
    }
}
