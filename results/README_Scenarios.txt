SCENARIOS: HISTORY-BASED EXTRAPOLATIONS TO 2030
Physics-Informed Extrapolation of Sectoral Fossil CO2 Emissions in the Energy Transition

WHAT THESE FILES ARE
These files are statistical extrapolations from the 2024 origin, not policy scenarios. They answer one question:
where would each country's fossil CO2 go by 2030 if the patterns of 1965-2024 continued? No announced policy,
nationally determined contribution, clean-energy deployment rate or fuel price enters the models. Use them as a
reference line that policy scenarios can be compared against, not as predictions (Section 4.7 of the manuscript).

THE THREE METHODS
  COMB-P    Main method of the paper. Equal-weight combination of the damped trend and XGBoost, followed by the
            post-forecast stoichiometric correction (CO2 = EF x E with each country's 2024 emission factor).
  XGB-P     Alternative. XGBoost with the post-forecast correction.
  damped-P  Alternative. Damped trend with the post-forecast correction.
The spread between the three is a rough measure of method uncertainty. For the 42-country panel (five sources):
  2024 observed   31.01 Gt CO2
  2030 COMB-P     34.06 Gt CO2  (+9.8%)
  2030 XGB-P      32.98 Gt CO2  (+6.3%)
  2030 damped-P   35.25 Gt CO2  (+13.7%)
COMB-P change 2024-2030 by source: coal +12.8%, oil +4.4%, gas +10.5%,
flaring +2.9%, cement +12.8%.

UNCERTAINTY
lo90 and hi90 are 90% split-conformal prediction intervals, calibrated on the same model's test-period forecasts
(origins 2008-2019) for the same source and horizon. In that test period the COMB-P intervals covered
88.0% of outcomes overall (89.6% one year ahead, 83.5% ten years ahead),
so they are somewhat too narrow at long horizons. The 'total' rows have their own interval, calibrated on errors of
the total; it is not the sum of the source intervals.

FIRST-YEAR CHECK
The Global Carbon Budget's preliminary estimates for 2025 put the change in fossil CO2 at +0.4% for China, +1.4%
for India and +1.9% for the United States. COMB-P projected +3.7%, +7.3% and −1.3%. All three
preliminary values fall inside the 90% intervals, but the central projections for China and India were too high
and the one for the United States too low.

FILES (one set of three per method, in results/)
  projections_<method>.csv         42 countries x 6 years x 6 series (5 sources + total)
  projections_panel_<method>.csv   42-country sums by source
  projections_energy_<method>.csv  matching coal, oil and gas consumption projections
Units: CO2 in Mt CO2 per year; energy in TWh per year. Every column is described in DATA_DICTIONARY.md.
The files are produced by src/project.py.
