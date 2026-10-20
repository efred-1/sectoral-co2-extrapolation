"""Build the energy-matched country panel and the stoichiometric diagnostic.

Inputs (both from Our World in Data):
  owid-co2-data.csv    -> Global Carbon Budget 2025 fossil CO2 by source (Mt)
  owid-energy-data.csv -> Energy Institute Statistical Review 2025 fuel consumption (TWh)

Outputs (data/):
  panel.csv            one row per country-year, 1965-2024, panel countries only
  panel_countries.csv  country list with coverage statistics
  ef_diagnostic.csv    implied emission factor per country-year-fuel with IPCC band class
"""
import sys
import numpy as np
import pandas as pd

DATA = "data"
SECTORS = ["coal_co2", "oil_co2", "gas_co2", "flaring_co2", "cement_co2"]
FUELS = ["coal", "oil", "gas"]
ENERGY = ["coal_consumption", "oil_consumption", "gas_consumption",
          "renewables_consumption", "nuclear_consumption", "primary_energy_consumption"]

# IPCC 2006 Guidelines Vol. 2 Ch. 1 Table 1.4 (kg CO2/TJ, 95% interval bounds), converted to t CO2/MWh.
KG_PER_TJ_TO_T_PER_MWH = 3.6e-6
IPCC_BAND = {
    # primary coals: coking coal / brown coal briquettes lower (87,300) to lignite upper (115,000)
    "coal": (87_300 * KG_PER_TJ_TO_T_PER_MWH, 115_000 * KG_PER_TJ_TO_T_PER_MWH),
    # combustible oil products: ethane lower (56,500) to residual fuel oil upper (78,800)
    "oil": (56_500 * KG_PER_TJ_TO_T_PER_MWH, 78_800 * KG_PER_TJ_TO_T_PER_MWH),
    # natural gas lower (54,300) to upper (58,300)
    "gas": (54_300 * KG_PER_TJ_TO_T_PER_MWH, 58_300 * KG_PER_TJ_TO_T_PER_MWH),
}
MIN_YEARS = 40          # of 55 possible years, 1970-2024
E_MIN_TWH = 1.0         # below this a fuel is treated as absent for emission-factor purposes
C_MIN_MT = 0.1


def main(co2_path, energy_path):
    C = pd.read_csv(co2_path)
    E = pd.read_csv(energy_path)
    C = C[C.iso_code.notna() & ~C.iso_code.str.startswith("OWID")]
    E = E[E.iso_code.notna() & ~E.iso_code.str.startswith("OWID")]
    world = pd.read_csv(co2_path).query("country == 'World'").set_index("year")["co2"]

    M = C[["country", "iso_code", "year", "population", "co2"] + SECTORS].merge(
        E[["iso_code", "year"] + ENERGY], on=["iso_code", "year"], how="left")
    M = M[M.year.between(1965, 2024)].sort_values(["country", "year"]).reset_index(drop=True)

    core = SECTORS + ["coal_consumption", "oil_consumption", "gas_consumption", "population"]
    ok = (M[core].notna().all(axis=1) & M.year.between(1970, 2024)).groupby(M.country).sum()
    keep = sorted(ok[ok >= MIN_YEARS].index)
    P = M[M.country.isin(keep)].copy()

    # Nuclear and renewables are blank in some EI rows where the country has none; treat as zero
    # only in years where the country's fossil-fuel rows are reported.
    reported = P[["coal_consumption", "oil_consumption", "gas_consumption"]].notna().all(axis=1)
    for c in ["renewables_consumption", "nuclear_consumption"]:
        P.loc[reported & P[c].isna(), c] = 0.0
    P["low_carbon_share"] = (P.renewables_consumption + P.nuclear_consumption) / P.primary_energy_consumption

    for f in FUELS:
        e, co2 = P[f"{f}_consumption"], P[f"{f}_co2"]
        valid = (e >= E_MIN_TWH) & (co2 >= C_MIN_MT)
        P[f"ef_{f}"] = np.where(valid, co2 / e, np.nan)   # t CO2 per MWh = Mt per TWh

    P.to_csv(f"{DATA}/panel.csv", index=False)

    # Country summary
    rows = []
    for c, g in P.groupby("country"):
        g24 = g[g.year == 2022]
        rows.append(dict(country=c, iso_code=g.iso_code.iloc[0], complete_years=int(ok[c]),
                         co2_2022_mt=float(g24.co2.iloc[0]) if len(g24) else np.nan))
    S = pd.DataFrame(rows).sort_values("co2_2022_mt", ascending=False)
    S["share_world_2022_pct"] = 100 * S.co2_2022_mt / world.loc[2022]
    S.to_csv(f"{DATA}/panel_countries.csv", index=False)

    # Stoichiometric diagnostic on the panel, 1970-2024
    D = []
    for f in FUELS:
        lo, hi = IPCC_BAND[f]
        sub = P[P.year.between(1970, 2024) & P[f"ef_{f}"].notna()][["country", "year", f"ef_{f}"]]
        sub = sub.rename(columns={f"ef_{f}": "ef"}).assign(fuel=f, band_lo=lo, band_hi=hi)
        sub["band_class"] = np.select([sub.ef > hi, sub.ef < lo], ["above_upper", "below_lower"], "within")
        D.append(sub)
    D = pd.concat(D, ignore_index=True)
    D.to_csv(f"{DATA}/ef_diagnostic.csv", index=False)

    print(f"panel: {len(keep)} countries, {len(P)} country-years (1965-2024)")
    print(f"share of world fossil CO2 2022: {S.share_world_2022_pct.sum():.1f}%")
    print("IPCC bands (t/MWh):", {k: (round(v[0], 4), round(v[1], 4)) for k, v in IPCC_BAND.items()})
    print(D.groupby(["fuel", "band_class"]).size().unstack(fill_value=0))
    print(D.groupby("fuel").ef.describe(percentiles=[.05, .5, .95]).round(3))
    print(S.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
