import unittest

import numpy as np

from Model2.Experiment1 import simulation as engines
from Model3.Experiment1 import simulation as model
from Model3.Experiment1.analysis import population_steps as study


class PopulationStepStudyTests(unittest.TestCase):
    def test_fast_kraus_matches_original_full_exponentiation(self):
        # N=7 checks the largest chain used in the study, including every Kraus element.
        for n in (3, 7):
            H, _, *_ = engines.build_model(n, exchange_prefactor=0.5)
            jumps = []
            for site, p in ((0, 0.9), (n-1, 0.1)):
                loss = engines.classical.operator_on_site(engines.classical.SIGMA_MINUS, site, n)
                jumps.extend((np.sqrt(2*p)*loss.getH(), np.sqrt(2*(1-p))*loss))
            for formula in ("lie", "strang"):
                expected = engines.dilation_kraus(H, jumps, 0.005, formula)
                actual = study.build_kraus(n, 0.005, formula)
                np.testing.assert_allclose(actual, expected, atol=3e-14)

    def test_sector_propagation_preserves_arbitrary_number_conserving_density(self):
        generator = np.random.default_rng(8134)
        for n in (3, 7):
            for formula in ("lie", "strang"):
                channel = study.BlockChannel(n, 0.005, formula)
                blocks = []
                for size in channel.sizes:
                    X = generator.normal(size=(size,size)) + 1j*generator.normal(size=(size,size))
                    blocks.append(X @ X.conj().T)
                normalization = sum(np.trace(b).real for b in blocks)
                blocks = [b/normalization for b in blocks]
                rho = np.zeros((2**n,2**n),complex)
                for sector, block in zip(channel.sectors,blocks):
                    rho[np.ix_(sector,sector)] = block
                expected = sum(K @ rho @ K.conj().T for K in study.build_kraus(n,0.005,formula))
                actual = np.zeros_like(rho)
                for sector, block in zip(channel.sectors,channel.apply(blocks)):
                    actual[np.ix_(sector,sector)] = block
                np.testing.assert_allclose(actual,expected,atol=2e-14)

    def test_batched_exact_populations_match_existing_correlations(self):
        times = np.asarray([0,0.17,0.6,2,20,120])
        for n in (3,7):
            labels = ("0"*n,"1"*n,"1"+"0"*(n-1))
            Z, _ = model.xx_correlations(times,labels)
            np.testing.assert_allclose(study.exact_populations(n,times),(1-Z)/2,atol=2e-14)


if __name__ == "__main__":
    unittest.main()
