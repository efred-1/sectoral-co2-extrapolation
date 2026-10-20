# Data dictionary

Every column of the cleaned data (`data/`) and of the projection files (`results/projections_*.csv`). Files are comma-separated, UTF-8, with one header row; a blank cell is a missing value. Ranges and missing counts are computed from the files by `src/make_data_dictionary.py`.

Units: CO₂ in million tonnes per year (Mt CO₂ yr⁻¹); energy in terawatt-hours per year (TWh yr⁻¹); emission factors in t CO₂ MWh⁻¹, which equals Mt CO₂ per TWh.

## `data/panel.csv` — Energy-matched country panel

One row per country and year, 1965–2024, for the 42 economies in the study. Fossil CO₂ by source is paired with fuel consumption for the same country-year. Countries are kept when all core variables (five CO₂ sources, coal, oil and gas consumption, population) are present in at least 40 of the 55 years 1970–2024. Aggregates (regions, income groups) were removed by keeping only entities with a country ISO code.

Rows: 2,520.

| Column | Unit | Description | Source / derivation | Range | Missing |
|---|---|---|---|---|---:|
| `country` | – | Country name as used by Our World in Data | Our World in Data | 42 distinct values | 0 |
| `iso_code` | – | ISO 3166-1 alpha-3 country code | Our World in Data | 42 distinct values | 0 |
| `year` | yr | Calendar year | – | 1965 to 2024 | 0 |
| `population` | persons | Population | Our World in Data population series (various sources, 2024) | 6.408e+04 to 1.451e+09 | 0 |
| `co2` | Mt CO₂ yr⁻¹ | Total fossil CO₂ emissions excluding land-use change. Includes the small 'other industry' category, so it can exceed the sum of the five source columns. The study's national total is the sum of the five sources, not this column | Global Carbon Budget 2025, via Our World in Data CO₂ dataset | 0.02565 to 1.229e+04 | 0 |
| `coal_co2` | Mt CO₂ yr⁻¹ | CO₂ from coal combustion | Global Carbon Budget 2025, via Our World in Data CO₂ dataset | 0 to 8886 | 61 |
| `oil_co2` | Mt CO₂ yr⁻¹ | CO₂ from oil combustion | Global Carbon Budget 2025, via Our World in Data CO₂ dataset | 0 to 2584 | 0 |
| `gas_co2` | Mt CO₂ yr⁻¹ | CO₂ from natural gas combustion | Global Carbon Budget 2025, via Our World in Data CO₂ dataset | 0 to 1748 | 36 |
| `flaring_co2` | Mt CO₂ yr⁻¹ | CO₂ from gas flaring | Global Carbon Budget 2025, via Our World in Data CO₂ dataset | 0 to 84.52 | 127 |
| `cement_co2` | Mt CO₂ yr⁻¹ | CO₂ from cement production (process emissions) | Global Carbon Budget 2025, via Our World in Data CO₂ dataset | 0 to 828.7 | 0 |
| `coal_consumption` | TWh yr⁻¹ | Primary energy consumption of coal | Energy Institute Statistical Review of World Energy 2025, via Our World in Data energy dataset | 0 to 2.56e+04 | 140 |
| `oil_consumption` | TWh yr⁻¹ | Primary energy consumption of oil | Energy Institute Statistical Review of World Energy 2025, via Our World in Data energy dataset | 0.097 to 1.122e+04 | 140 |
| `gas_consumption` | TWh yr⁻¹ | Primary energy consumption of natural gas | Energy Institute Statistical Review of World Energy 2025, via Our World in Data energy dataset | 0 to 9022 | 140 |
| `renewables_consumption` | TWh yr⁻¹ | Primary energy from renewables, substitution method. Set to 0 where blank in a year in which the country's fossil-fuel rows are reported | Energy Institute Statistical Review of World Energy 2025, via Our World in Data energy dataset | 0 to 8470 | 140 |
| `nuclear_consumption` | TWh yr⁻¹ | Primary energy from nuclear, substitution method. Set to 0 where blank in a year in which the country's fossil-fuel rows are reported | Energy Institute Statistical Review of World Energy 2025, via Our World in Data energy dataset | 0 to 2258 | 140 |
| `primary_energy_consumption` | TWh yr⁻¹ | Total primary energy consumption | Our World in Data series combining Energy Institute and U.S. EIA data | 0.097 to 4.899e+04 | 140 |
| `low_carbon_share` | fraction (0–1) | (renewables_consumption + nuclear_consumption) / primary_energy_consumption | Derived in src/build_data.py | 0 to 0.6883 | 140 |
| `ef_coal` | t CO₂ MWh⁻¹ | Implied emission factor of coal, coal_co2 / coal_consumption. Blank where consumption < 1 TWh or CO₂ < 0.1 Mt. Values outside the IPCC range are kept as reported (Section 2.3 of the manuscript) | Derived in src/build_data.py (Eq. 1 of the manuscript) | 0.1114 to 0.7646 | 499 |
| `ef_oil` | t CO₂ MWh⁻¹ | Implied emission factor of oil, oil_co2 / oil_consumption. Same validity rule. Values outside the IPCC range are kept as reported; the maximum, Kuwait 1991, most likely reflects the oil-well fires of that year, which the inventory counts as oil CO₂ | Derived in src/build_data.py (Eq. 1) | 0.005151 to 11.23 | 150 |
| `ef_gas` | t CO₂ MWh⁻¹ | Implied emission factor of gas, gas_co2 / gas_consumption. Same validity rule | Derived in src/build_data.py (Eq. 1) | 0.004009 to 1.193 | 254 |

## `data/panel_countries.csv` — Panel countries

The 42 economies in the panel, sorted by fossil CO₂ in 2022 (Table A.1 of the manuscript).

Rows: 42.

| Column | Unit | Description | Source / derivation | Range | Missing |
|---|---|---|---|---|---:|
| `country` | – | Country name | Our World in Data | 42 distinct values | 0 |
| `iso_code` | – | ISO 3166-1 alpha-3 country code | Our World in Data | 42 distinct values | 0 |
| `complete_years` | yr | Number of years in 1970–2024 (of 55) with every core variable present | Derived in src/build_data.py | 40 to 55 | 0 |
| `co2_2022_mt` | Mt CO₂ | Total fossil CO₂ in 2022 (column co2 of the panel file) | Global Carbon Budget 2025, via Our World in Data CO₂ dataset | 6.608 to 1.171e+04 | 0 |
| `share_world_2022_pct` | % | Share of world fossil CO₂ in 2022 | Derived in src/build_data.py | 0.01761 to 31.21 | 0 |

## `data/ef_diagnostic.csv` — Emission-factor audit

Each valid implied emission factor, 1970–2024, compared with the IPCC 2006 default range (95% interval bounds of Vol. 2, Ch. 1, Table 1.4, converted from kg CO₂/TJ by × 3.6 × 10⁻⁶). Coal: coking coal lower bound to lignite upper bound; oil: ethane lower to residual fuel oil upper; gas: natural gas lower to upper. Basis of Fig. 3 and Section 2.3 of the manuscript.

Rows: 6,233.

| Column | Unit | Description | Source / derivation | Range | Missing |
|---|---|---|---|---|---:|
| `country` | – | Country name | – | 42 distinct values | 0 |
| `year` | yr | Calendar year | – | 1970 to 2024 | 0 |
| `ef` | t CO₂ MWh⁻¹ | Implied emission factor (as in ef_coal, ef_oil and ef_gas of the panel file) | Derived in src/build_data.py | 0.004009 to 11.23 | 0 |
| `fuel` | – | coal, oil or gas | – | 3 distinct values | 0 |
| `band_lo` | t CO₂ MWh⁻¹ | Lower bound of the IPCC 2006 range for the fuel | IPCC 2006 Guidelines | 0.1955 to 0.3143 | 0 |
| `band_hi` | t CO₂ MWh⁻¹ | Upper bound of the IPCC 2006 range for the fuel. A value above it means more CO₂ is reported than the reported fuel could release | IPCC 2006 Guidelines | 0.2099 to 0.414 | 0 |
| `band_class` | – | within, above_upper or below_lower | Derived in src/build_data.py | 3 distinct values | 0 |

## `results/projections_COMB-P.csv`, `results/projections_XGB-P.csv`, `results/projections_damped-P.csv`

Projections by country and source from origin 2024, horizons 1–6 years, for the three methods: COMB-P (the paper's main method), XGB-P and damped-P. These are extrapolations of history, not policy scenarios. Intervals are calibrated on the same model's test-period forecasts (origins 2008–2019) for the same source and horizon.

Rows per file: 1,512. Ranges below are for the COMB-P file.

| Column | Unit | Description | Source / derivation | Range | Missing |
|---|---|---|---|---|---:|
| `country` | – | Country name | – | 42 distinct values | 0 |
| `year` | yr | Target year, 2025–2030 | – | 2025 to 2030 | 0 |
| `h` | yr | Horizon from the 2024 origin, 1–6 | – | 1 to 6 | 0 |
| `source` | – | coal, oil, gas, flaring, cement, or total (sum of the five) | – | 6 distinct values | 0 |
| `proj` | Mt CO₂ yr⁻¹ | Point projection | src/project.py | 0 to 1.382e+04 | 0 |
| `lo90` | Mt CO₂ yr⁻¹ | Lower bound of the 90% split-conformal prediction interval (floored at 0) | src/project.py | 0 to 1.124e+04 | 0 |
| `hi90` | Mt CO₂ yr⁻¹ | Upper bound of the 90% split-conformal prediction interval | src/project.py | 0.01898 to 1.802e+04 | 0 |
| `base_2024` | Mt CO₂ yr⁻¹ | Observed value in 2024, the origin year | Global Carbon Budget 2025, via Our World in Data CO₂ dataset | 0 to 1.21e+04 | 0 |

## `results/projections_panel_COMB-P.csv`, `results/projections_panel_XGB-P.csv`, `results/projections_panel_damped-P.csv`

Sums over the 42 countries of the source projections, one file per method.

Rows per file: 30. Ranges below are for the COMB-P file.

| Column | Unit | Description | Source / derivation | Range | Missing |
|---|---|---|---|---|---:|
| `year` | yr | Target year, 2025–2030 | – | 2025 to 2030 | 0 |
| `source` | – | coal, oil, gas, flaring or cement | – | 5 distinct values | 0 |
| `proj` | Mt CO₂ yr⁻¹ | Projection summed over the 42 countries | src/project.py | 354 to 1.59e+04 | 0 |
| `base_2024` | Mt CO₂ yr⁻¹ | Observed 2024 value summed over the 42 countries | Global Carbon Budget 2025, via Our World in Data CO₂ dataset | 351.5 to 1.409e+04 | 0 |

## `results/projections_energy_COMB-P.csv`, `results/projections_energy_XGB-P.csv`, `results/projections_energy_damped-P.csv`

Fuel-energy projections that go with each method's CO₂ projections. They satisfy CO₂ = EF × E with the 2024 emission factor wherever the fuel is present.

Rows per file: 252. Ranges below are for the COMB-P file.

| Column | Unit | Description | Source / derivation | Range | Missing |
|---|---|---|---|---|---:|
| `country` | – | Country name | – | 42 distinct values | 0 |
| `year` | yr | Target year, 2025–2030 | – | 2025 to 2030 | 0 |
| `coal_twh` | TWh yr⁻¹ | Projected coal consumption, after the post-forecast correction | src/project.py | 0 to 2.912e+04 | 0 |
| `oil_twh` | TWh yr⁻¹ | Projected oil consumption, after the post-forecast correction | src/project.py | 15.19 to 9877 | 0 |
| `gas_twh` | TWh yr⁻¹ | Projected gas consumption, after the post-forecast correction | src/project.py | 3.439 to 9345 | 0 |
