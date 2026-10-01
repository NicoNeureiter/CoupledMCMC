package coupledMCMC;

import org.apache.commons.statistics.distribution.BetaDistribution;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class CoupledPathSamplerTest {

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
