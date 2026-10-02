# Coupled MCMC (MC3) for BEAST

Parallel tempering (Metropolis-coupled MCMC) for [BEAST 3](https://github.com/CompEvol/beast3).

## Maven coordinates

```xml
<dependency>
    <groupId>io.github.compevol</groupId>
    <artifactId>coupled-mcmc</artifactId>
    <version>1.3.0-beta1</version>
</dependency>
```

JPMS module: `coupled.mcmc`

## How to set up your analysis to run with coupled MCMC

### Using the conversion app

After installing the `CoupledMCMC` package, the `MCMC2CoupledMCMC` app becomes available in the app launcher.

1. Create an MCMC analysis in BEAUti with any of the available templates, save as `mcmc.xml`

2. From a terminal, run:

   ```
   /path/to/beast/bin/applauncher MCMC2CoupledMCMC -xml mcmc.xml -o mc3.xml
   ```

   This creates a file `mc3.xml` containing a CoupledMCMC analysis with the same model/operators/loggers as the `mcmc.xml` analysis.

   Alternatively, from BEAUti use menu `File > Launch apps`, select `MCMC to Coupled MCMC converter` from the available apps, fill in the form and click OK.

### By editing an XML file

In a pre-prepared XML, replace the MCMC run element:

```xml
<run id="mcmc" spec="MCMC" chainLength="....." numInitializationAttempts="....">
```

with:

```xml
<run id="mcmc" spec="coupledMCMC.CoupledMCMC" chainLength="100000000" storeEvery="1000000" deltaTemperature="0.025" chains="2" resampleEvery="10000">
```

* `deltaTemperature="0.025"` — temperature difference between chain *n* and chain *n-1*. Tune so that the swap acceptance probability is between 0.25 and 0.6.
* `chains="2"` — number of parallel chains. The first (cold) chain explores the posterior; heated chains have higher acceptance probabilities and propose new states to the cold chain.

## Coupled path sampling

`CoupledPathSampler` estimates the marginal likelihood by path sampling / stepping stone (as `PathSampler` in the model-selection package), but runs the power-posterior steps as coupled chains, so that neighbouring steps can swap states. The posterior must be a `CompoundDistribution` with a distribution of id `likelihood`; everything else in it counts as prior.

```xml
<run id="mcmc" spec="coupledMCMC.CoupledPathSampler" chainLength="1000000" storeEvery="10000"
     chains="16" alpha="0.3" rootdir="cps" burnInPercentage="50" resampleEvery="100">
```

* `chains` — number of steps. Step *k* samples prior × likelihood<sup>β<sub>k</sub></sup>, with β running from 1 (step 0) to 0 (the prior), spaced as quantiles of Beta(`alpha`, 1); `alpha="0"` gives uniform spacing.
* `chainLength` — samples per step. All steps run at the same time, one thread each, so the run needs memory and cores for `chains` copies of the model.
* `burnInPercentage` — discarded from each step's log. Steps start cold rather than from the previous step's state, so this has to cover burn-in.
* `heatLikelihoodOnly` and `optimise` are fixed to `true` and `false`; `neighbourSwapping` defaults to `true`.
* `evenOddSwaps` (default `true`) — every `resampleEvery` samples, attempt all even neighbour pairs (0,1),(2,3),… or all odd pairs (1,2),(3,4),…, alternating, instead of one random neighbour pair. States then cross the ladder in O(`chains`) rather than O(`chains`²) rounds (Syed et al. 2021).

Each step's likelihood trace is written to `<rootdir>/step<k>/likelihood.log`, the same layout `PathSampler` uses, so `PathSampleAnalyser` can re-analyse it. At the end the per-step table and the line `marginal L estimate = ...` are printed (stepping stone; thermodynamic integration when `alpha <= 0`). Swaps correlate the steps, so standard errors computed as if the steps were independent are too small. See `examples/coupled_path_sampling.xml`.

### Batched coupled path sampling

`BatchedCoupledPathSampler` runs the steps in sequential batches instead of all at once. The steps of a batch are coupled chains, as above, and every chain of the next batch starts from the final state (and operator tuning) of the previous batch's last step, the one with the smallest beta, instead of from the initial state. Only the first batch has to burn in from scratch, so the chains can be much shorter and the number of steps larger.

```xml
<run id="mcmc" spec="coupledMCMC.BatchedCoupledPathSampler" chainLength="500000" preBurnin="1500000"
     chains="64" batchSize="8" alpha="0.25" rootdir="cps" burnInPercentage="20" resampleEvery="100">
```

* `chains` — total number of steps; `batchSize` — how many run at the same time, for example the number of CPU cores. The last batch takes the remainder. Without `batchSize` all steps run at once and this is a plain `CoupledPathSampler`.
* `preBurnin` — samples the first batch runs before it logs anything. It is the only batch that starts from the initial state, so size it for the burn-in of a cold start.
* Swaps happen within a batch only, and the batches follow the steps from beta = 1 to beta = 0. Output files and the estimate are the same as for `CoupledPathSampler`. Runs cannot be resumed.

## Citation

Müller NF, Bouckaert RR. Adaptive Metropolis-coupled MCMC for BEAST 2. PeerJ. 2020 Sep 16;8:e9473. [DOI:10.7717/peerj.9473](https://doi.org/10.7717/peerj.9473)

Altekar G, Dwarkadas S, Huelsenbeck JP, Ronquist F. Parallel Metropolis coupled Markov chain Monte Carlo for Bayesian phylogenetic inference. [Bioinformatics 20(3):407–415](https://academic.oup.com/bioinformatics/article/20/3/407/186341)
