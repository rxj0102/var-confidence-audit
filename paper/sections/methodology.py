"""Methodology section: backtesting framework and model derivations."""

from __future__ import annotations

from reportlab.platypus import Paragraph

SETUP = (
    "Let r<sub>t</sub> denote the bank's daily log return (the P&amp;L proxy) "
    "and V<sub>t</sub> the disclosed one-day VaR for the quarter containing "
    "day t, normalized into return space as described in Section 2. Define "
    "the violation indicator I<sub>t</sub> = 1{|r<sub>t</sub>| &gt; "
    "V<sub>t</sub>}. If the bank's stated confidence level is 1&minus;p "
    "(p = 0.01 for 99% VaR, p = 0.05 for 95% VaR), correct calibration "
    "requires the hit sequence {I<sub>t</sub>} to be i.i.d. Bernoulli(p): "
    "violations should occur at rate p (correct coverage) and should not "
    "cluster (independence). The tests below examine each property and "
    "their conjunction."
)

KUPIEC_HEAD = "3.1 Kupiec (1995) proportion-of-failures test"
KUPIEC_BODY = (
    "With T observations and T<sub>1</sub> violations, the likelihood under "
    "the null violation probability p is p<sup>T1</sup>(1&minus;p)"
    "<sup>T&minus;T1</sup>, while the unconstrained maximum-likelihood "
    "estimate is the observed frequency &pi;&#770; = T<sub>1</sub>/T. The "
    "likelihood-ratio statistic is"
)
KUPIEC_EQ = (
    "LR<sub>POF</sub> = &minus;2 ln[ p<sup>T1</sup> (1&minus;p)"
    "<sup>T&minus;T1</sup> / ( &pi;&#770;<sup>T1</sup> "
    "(1&minus;&pi;&#770;)<sup>T&minus;T1</sup> ) ]  ~  &chi;&sup2;(1)."
)
KUPIEC_TAIL = (
    "Under H<sub>0</sub> the statistic is asymptotically chi-squared with one "
    "degree of freedom; the test is two-tailed in the violation count, "
    "rejecting both models that understate risk (too many violations) and "
    "models that are excessively conservative (too few). Inverting the test "
    "at the 5% level yields a non-rejection interval [T<sub>1</sub><sup>lo"
    "</sup>, T<sub>1</sub><sup>hi</sup>] for the acceptable number of "
    "violations given T; banks falling outside the interval are flagged. The "
    "test is computed per bank for the full sample, each calendar year, and "
    "each VIX regime (low &le; 15, medium 15&ndash;25, high &gt; 25), at the "
    "1%, 5%, and 10% significance levels."
)

CHRIST_HEAD = "3.2 Christoffersen (1998) three-component framework"
CHRIST_UC = (
    "<b>Unconditional coverage (UC).</b> Identical in form to the Kupiec "
    "statistic: with n<sub>1</sub> violations and n<sub>0</sub> "
    "non-violations, LR<sub>UC</sub> = &minus;2 ln[ p<sup>n1</sup>"
    "(1&minus;p)<sup>n0</sup> / ( p&#770;<sup>n1</sup>(1&minus;p&#770;)"
    "<sup>n0</sup> ) ] ~ &chi;&sup2;(1), where p&#770; = n<sub>1</sub>/"
    "(n<sub>0</sub>+n<sub>1</sub>)."
)
CHRIST_IND = (
    "<b>Independence (IND).</b> Model the hit sequence as a first-order "
    "Markov chain with transition probabilities &pi;<sub>01</sub> = "
    "P(I<sub>t</sub>=1 | I<sub>t&minus;1</sub>=0) and &pi;<sub>11</sub> = "
    "P(I<sub>t</sub>=1 | I<sub>t&minus;1</sub>=1), estimated from the "
    "transition counts n<sub>ij</sub>. Under independence &pi;<sub>01</sub> "
    "= &pi;<sub>11</sub>; the restricted likelihood uses the pooled hit rate "
    "&pi;&#770;. The statistic LR<sub>IND</sub> = &minus;2 ln[ "
    "L(&pi;&#770;) / L(&pi;&#770;<sub>01</sub>, &pi;&#770;<sub>11</sub>) ] "
    "~ &chi;&sup2;(1) rejects when violations today predict violations "
    "tomorrow — the clustering signature of a model that fails to update "
    "in volatile markets."
)
CHRIST_CC = (
    "<b>Conditional coverage (CC).</b> Because the two components are "
    "asymptotically independent, the joint test of correct coverage and "
    "independence is simply LR<sub>CC</sub> = LR<sub>UC</sub> + "
    "LR<sub>IND</sub> ~ &chi;&sup2;(2). A model passes the audit only if it "
    "survives the joint test. All three statistics are computed per bank for "
    "the full sample and the stress / calm subsamples, with rejection "
    "assessed at the 5% level. Degenerate cells (zero counts) follow the "
    "convention 0&middot;ln 0 = 0."
)

MODELS_HEAD = "3.3 Model reimplementation"
MODELS_BODY = (
    "To benchmark the disclosed figures, three textbook 99% one-day VaR "
    "models are estimated on the same return series. (i) <i>Historical "
    "simulation</i>: the 99th percentile of the empirical loss distribution "
    "over a rolling 250-day window, with no distributional assumption. (ii) "
    "<i>Parametric variance-covariance</i>: VaR<sub>t</sub> = &minus;("
    "&mu;&#770;<sub>t</sub> &minus; z<sub>0.99</sub>&sigma;&#770;<sub>t</sub>) "
    "with z<sub>0.99</sub> = 2.326 and rolling 250-day moments; a Student-t "
    "variant re-estimates the degrees of freedom by maximum likelihood "
    "(refit monthly) and scales the quantile so the innovation variance "
    "matches &sigma;&#770;&sup2;<sub>t</sub>. (iii) <i>EWMA</i> "
    "(RiskMetrics): &sigma;&sup2;<sub>t</sub> = &lambda;&sigma;&sup2;"
    "<sub>t&minus;1</sub> + (1&minus;&lambda;)r&sup2;<sub>t&minus;1</sub> "
    "with &lambda; = 0.94 and VaR<sub>t</sub> = z<sub>0.99</sub>&sigma;"
    "<sub>t</sub>. All forecasts are strictly out-of-sample (the estimation "
    "window precedes the forecast day), are converted to dollars with the "
    "same trading-equity base used for the disclosures, and are subjected "
    "to the identical violation definition and Kupiec test."
)

STRESS_HEAD = "3.4 Stress classification and comparisons"
STRESS_BODY = (
    "Days with VIX &gt; 25 form the stress subsample. For each bank the "
    "stress amplification factor is the ratio of the stress-period violation "
    "rate to the calm-period rate; a correctly specified model keeps this "
    "ratio near one because its VaR expands with volatility. Cross-bank "
    "rankings combine the excess violation rate, the conditional-coverage "
    "p-value, and consistency (the standard deviation of annual violation "
    "rates). Finally, banks are grouped by disclosed methodology "
    "(historical simulation vs Monte Carlo vs parametric) and a one-way "
    "ANOVA tests whether methodology explains differences in quarterly "
    "violation rates."
)


def build(styles) -> list:
    """Build the methodology section flowables.

    Args:
        styles: Paper stylesheet.

    Returns:
        List of ReportLab flowables.
    """
    return [
        Paragraph("3. Methodology", styles["Heading"]),
        Paragraph(SETUP, styles["Body"]),
        Paragraph(KUPIEC_HEAD, styles["SubHeading"]),
        Paragraph(KUPIEC_BODY, styles["Body"]),
        Paragraph(KUPIEC_EQ, styles["Equation"]),
        Paragraph(KUPIEC_TAIL, styles["Body"]),
        Paragraph(CHRIST_HEAD, styles["SubHeading"]),
        Paragraph(CHRIST_UC, styles["Body"]),
        Paragraph(CHRIST_IND, styles["Body"]),
        Paragraph(CHRIST_CC, styles["Body"]),
        Paragraph(MODELS_HEAD, styles["SubHeading"]),
        Paragraph(MODELS_BODY, styles["Body"]),
        Paragraph(STRESS_HEAD, styles["SubHeading"]),
        Paragraph(STRESS_BODY, styles["Body"]),
    ]
