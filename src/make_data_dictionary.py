"""Data dictionary for the cleaned data and the projection (extrapolation) files.

Writes results/data_dictionary.csv (one row per column), results/data_dictionary.json (used for the Word
version in the submission package) and DATA_DICTIONARY.md (readable on GitHub, with the paths used in this code). Row counts, ranges and missing-value counts are read
from the files themselves, so the dictionary cannot drift from the data.
"""
import json
import pandas as pd

GCB = "Global Carbon Budget 2025, via Our World in Data CO₂ dataset"
EI = "Energy Institute Statistical Review of World Energy 2025, via Our World in Data energy dataset"
DER = "Derived in src/build_data.py"
MT = "Mt CO₂ yr⁻¹"
TWH = "TWh yr⁻¹"

FILES = [
    dict(file="energy_matched_panel_42countries_1965-2024.csv", folder="03_Cleaned_data", path="cleaned_data/energy_matched_panel_42countries_1965-2024.csv",
         title="Energy-matched country panel",
         about="One row per country and year, 1965–2024, for the 42 economies in the study. Fossil CO₂ by source "
               "is paired with fuel consumption for the same country-year. Countries are kept when all core "
               "variables (five CO₂ sources, coal, oil and gas consumption, population) are present in at least "
               "40 of the 55 years 1970–2024. Aggregates (regions, income groups) were removed by keeping only "
               "entities with a country ISO code.",
         cols=[
             ("country", "text", "–", "Country name as used by Our World in Data", "Our World in Data"),
             ("iso_code", "text", "–", "ISO 3166-1 alpha-3 country code", "Our World in Data"),
             ("year", "integer", "yr", "Calendar year", "–"),
             ("population", "number", "persons", "Population", "Our World in Data population series (various sources, 2024)"),
             ("co2", "number", MT, "Total fossil CO₂ emissions excluding land-use change. Includes the small "
              "'other industry' category, so it can exceed the sum of the five source columns. The study's "
              "national total is the sum of the five sources, not this column", GCB),
             ("coal_co2", "number", MT, "CO₂ from coal combustion", GCB),
             ("oil_co2", "number", MT, "CO₂ from oil combustion", GCB),
             ("gas_co2", "number", MT, "CO₂ from natural gas combustion", GCB),
             ("flaring_co2", "number", MT, "CO₂ from gas flaring", GCB),
             ("cement_co2", "number", MT, "CO₂ from cement production (process emissions)", GCB),
             ("coal_consumption", "number", TWH, "Primary energy consumption of coal", EI),
             ("oil_consumption", "number", TWH, "Primary energy consumption of oil", EI),
             ("gas_consumption", "number", TWH, "Primary energy consumption of natural gas", EI),
             ("renewables_consumption", "number", TWH, "Primary energy from renewables, substitution method. Set to 0 "
              "where blank in a year in which the country's fossil-fuel rows are reported", EI),
             ("nuclear_consumption", "number", TWH, "Primary energy from nuclear, substitution method. Set to 0 where "
              "blank in a year in which the country's fossil-fuel rows are reported", EI),
             ("primary_energy_consumption", "number", TWH, "Total primary energy consumption",
              "Our World in Data series combining Energy Institute and U.S. EIA data"),
             ("low_carbon_share", "number", "fraction (0–1)", "(renewables_consumption + nuclear_consumption) / "
              "primary_energy_consumption", DER),
             ("ef_coal", "number", "t CO₂ MWh⁻¹", "Implied emission factor of coal, coal_co2 / coal_consumption. "
              "Blank where consumption < 1 TWh or CO₂ < 0.1 Mt. Values outside the IPCC range are kept as reported "
              "(Section 2.3 of the manuscript)", DER + " (Eq. 1 of the manuscript)"),
             ("ef_oil", "number", "t CO₂ MWh⁻¹", "Implied emission factor of oil, oil_co2 / oil_consumption. Same "
              "validity rule. Values outside the IPCC range are kept as reported; the maximum, Kuwait 1991, most "
              "likely reflects the oil-well fires of that year, which the inventory counts as oil CO₂", DER + " (Eq. 1)"),
             ("ef_gas", "number", "t CO₂ MWh⁻¹", "Implied emission factor of gas, gas_co2 / gas_consumption. Same "
              "validity rule", DER + " (Eq. 1)"),
         ]),
    dict(file="panel_countries.csv", folder="03_Cleaned_data", path="cleaned_data/panel_countries.csv",
         title="Panel countries",
         about="The 42 economies in the panel, sorted by fossil CO₂ in 2022 (Table A.1 of the manuscript).",
         cols=[
             ("country", "text", "–", "Country name", "Our World in Data"),
             ("iso_code", "text", "–", "ISO 3166-1 alpha-3 country code", "Our World in Data"),
             ("complete_years", "integer", "yr", "Number of years in 1970–2024 (of 55) with every core variable present", DER),
             ("co2_2022_mt", "number", "Mt CO₂", "Total fossil CO₂ in 2022 (column co2 of the panel file)", GCB),
             ("share_world_2022_pct", "number", "%", "Share of world fossil CO₂ in 2022", DER),
         ]),
    dict(file="emission_factor_audit.csv", folder="03_Cleaned_data", path="cleaned_data/emission_factor_audit.csv",
         title="Emission-factor audit",
         about="Each valid implied emission factor, 1970–2024, compared with the IPCC 2006 default range "
               "(95% interval bounds of Vol. 2, Ch. 1, Table 1.4, converted from kg CO₂/TJ by × 3.6 × 10⁻⁶). "
               "Coal: coking coal lower bound to lignite upper bound; oil: ethane lower to residual fuel oil upper; "
               "gas: natural gas lower to upper. Basis of Fig. 3 and Section 2.3 of the manuscript.",
         cols=[
             ("country", "text", "–", "Country name", "–"),
             ("year", "integer", "yr", "Calendar year", "–"),
             ("ef", "number", "t CO₂ MWh⁻¹", "Implied emission factor (as in ef_coal, ef_oil and ef_gas of the panel file)", DER),
             ("fuel", "text", "–", "coal, oil or gas", "–"),
             ("band_lo", "number", "t CO₂ MWh⁻¹", "Lower bound of the IPCC 2006 range for the fuel", "IPCC 2006 Guidelines"),
             ("band_hi", "number", "t CO₂ MWh⁻¹", "Upper bound of the IPCC 2006 range for the fuel. A value above it "
              "means more CO₂ is reported than the reported fuel could release", "IPCC 2006 Guidelines"),
             ("band_class", "text", "–", "within, above_upper or below_lower", DER),
         ]),
]

PROJ_COLS = [
    ("country", "text", "–", "Country name", "–"),
    ("year", "integer", "yr", "Target year, 2025–2030", "–"),
    ("h", "integer", "yr", "Horizon from the 2024 origin, 1–6", "–"),
    ("source", "text", "–", "coal, oil, gas, flaring, cement, or total (sum of the five)", "–"),
    ("proj", "number", MT, "Point projection", "src/project.py"),
    ("lo90", "number", MT, "Lower bound of the 90% split-conformal prediction interval (floored at 0)", "src/project.py"),
    ("hi90", "number", MT, "Upper bound of the 90% split-conformal prediction interval", "src/project.py"),
    ("base_2024", "number", MT, "Observed value in 2024, the origin year", GCB),
]
PANEL_COLS = [
    ("year", "integer", "yr", "Target year, 2025–2030", "–"),
    ("source", "text", "–", "coal, oil, gas, flaring or cement", "–"),
    ("proj", "number", MT, "Projection summed over the 42 countries", "src/project.py"),
    ("base_2024", "number", MT, "Observed 2024 value summed over the 42 countries", GCB),
]
ENERGY_COLS = [
    ("country", "text", "–", "Country name", "–"),
    ("year", "integer", "yr", "Target year, 2025–2030", "–"),
    ("coal_twh", "number", TWH, "Projected coal consumption, after the post-forecast correction", "src/project.py"),
    ("oil_twh", "number", TWH, "Projected oil consumption, after the post-forecast correction", "src/project.py"),
    ("gas_twh", "number", TWH, "Projected gas consumption, after the post-forecast correction", "src/project.py"),
]
MODELS = [("COMB-P", "main method of the paper: equal-weight combination of the damped trend and XGBoost, with the "
                     "post-forecast stoichiometric correction"),
          ("XGB-P", "alternative: XGBoost with the post-forecast correction"),
          ("damped-P", "alternative: damped trend with the post-forecast correction")]
for m, desc in MODELS:
    FILES.append(dict(file=f"Extrapolation_{m}_country_by_source_2025-2030.csv", folder="08_Scenarios",
                      path=f"results/projections_{m}.csv", title=f"{m} extrapolation by country and source",
                      about=f"{m}: {desc}. Origin 2024, horizons 1–6 years. Intervals are calibrated on the "
                            "same model's test-period forecasts (origins 2008–2019) for the same source and horizon.",
                      cols=PROJ_COLS))
    FILES.append(dict(file=f"Extrapolation_{m}_panel_by_source_2025-2030.csv", folder="08_Scenarios",
                      path=f"results/projections_panel_{m}.csv", title=f"{m} extrapolation, 42-country panel totals",
                      about=f"Sum over the 42 countries of the {m} source projections.", cols=PANEL_COLS))
    FILES.append(dict(file=f"Extrapolation_{m}_fuel_energy_2025-2030.csv", folder="08_Scenarios",
                      path=f"results/projections_energy_{m}.csv", title=f"{m} fuel-energy extrapolation",
                      about=f"Fuel-energy projections that go with the {m} CO₂ projections. They satisfy "
                            "CO₂ = EF × E with the 2024 emission factor wherever the fuel is present.",
                      cols=ENERGY_COLS))

rows, out = [], []
for F in FILES:
    D = pd.read_csv(F["path"])
    assert list(D.columns) == [c[0] for c in F["cols"]], (F["file"], list(D.columns))
    F["n_rows"] = len(D)
    cols = []
    for name, typ, unit, desc, src in F["cols"]:
        s = D[name]
        if typ == "text":
            rng = f"{s.nunique()} distinct values"
        else:
            lo, hi = s.min(), s.max()
            rng = f"{lo:g} to {hi:g}" if typ == "integer" else f"{lo:.4g} to {hi:.4g}"
        miss = int(s.isna().sum())
        cols.append(dict(column=name, type=typ, unit=unit, description=desc, source=src, range=rng, missing=miss))
        rows.append(dict(folder=F["folder"], file=F["file"], column=name, type=typ, unit=unit, description=desc,
                         source=src, observed_range=rng, missing_values=miss, rows_in_file=len(D)))
    out.append(dict(file=F["file"], folder=F["folder"], path=F["path"], title=F["title"], about=F["about"],
                    n_rows=len(D), cols=cols))
pd.DataFrame(rows).to_csv("results/data_dictionary.csv", index=False)
json.dump(out, open("results/data_dictionary.json", "w"), indent=1, ensure_ascii=False)

# Markdown version for the code repository. Paths are the ones the code uses; the submission package renames a few files.
md = ["# Data dictionary", "",
      "Every column of the cleaned data (`cleaned_data/`) and of the projection files (`results/projections_*.csv`). "
      "Files are comma-separated, UTF-8, with one header row; a blank cell is a missing value. Ranges and missing "
      "counts are computed from the files by `src/make_data_dictionary.py`.", "",
      "Units: CO₂ in million tonnes per year (Mt CO₂ yr⁻¹); energy in terawatt-hours per year (TWh yr⁻¹); emission "
      "factors in t CO₂ MWh⁻¹, which equals Mt CO₂ per TWh.", ""]
seen = set()
for F in out:
    key = tuple(c["column"] for c in F["cols"]) if F["folder"] == "08_Scenarios" else F["path"]
    if F["folder"] == "08_Scenarios":
        if key in seen:
            continue
        seen.add(key)
        paths = [G["path"] for G in out if G["folder"] == "08_Scenarios" and tuple(c["column"] for c in G["cols"]) == key]
        kind = "energy" if "energy" in F["path"] else "panel" if "panel" in F["path"] else "country"
        about = {"country": "Projections by country and source from origin 2024, horizons 1–6 years, for the three "
                            "methods: COMB-P (the paper's main method), XGB-P and damped-P. These are extrapolations "
                            "of history, not policy scenarios. Intervals are calibrated on the same model's "
                            "test-period forecasts (origins 2008–2019) for the same source and horizon.",
                 "panel": "Sums over the 42 countries of the source projections, one file per method.",
                 "energy": "Fuel-energy projections that go with each method's CO₂ projections. They satisfy "
                           "CO₂ = EF × E with the 2024 emission factor wherever the fuel is present."}[kind]
        md += [f"## {', '.join('`' + q + '`' for q in paths)}", "", about, "",
               f"Rows per file: {F['n_rows']:,}. Ranges below are for the COMB-P file.", ""]
    else:
        md += [f"## `{F['path']}` — {F['title']}", "", F["about"], "", f"Rows: {F['n_rows']:,}.", ""]
    md += ["| Column | Unit | Description | Source / derivation | Range | Missing |", "|---|---|---|---|---|---:|"]
    for c in F["cols"]:
        md.append(f"| `{c['column']}` | {c['unit']} | {c['description']} | {c['source']} | {c['range']} | {c['missing']} |")
    md.append("")
open("DATA_DICTIONARY.md", "w").write("\n".join(md))
print(f"{len(rows)} columns in {len(out)} files")
