import pandas as pd
import numpy as np

print("Importing data")
DATA_FILE = "500000 teds/teds_d_2017_1"
data = pd.read_csv(DATA_FILE)

#remove "cheating" columns
data = data.drop(columns=["DISYR", "CASEID", "CBSA2010", "EMPLOY_D", "DETNLF", "DETNLF_D", "LIVARAG_D", "ARRESTS_D",
                   "SERVICES_D", "REASON", "DETCRIM", "SUB1_D", "FREQ1_D", "SUB2_D", "FREQ2_D", "SUB3_D", "FREQ3_D",
                    "FRSTUSE3", "HLTHINS", "PRIMPAY", "FREQ_ATND_SELF_HELP_D"])

#remove one-hotted columns
reduced_data = data.drop(columns=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY", "PREG", "VET", "LIVARAG", "PRIMINC",
                                  "STFIPS", "REGION", "DIVISION", "SERVICES", "METHUSE", "PSOURCE", "SUB1", "ROUTE1",
                                  "FREQ1", "SUB2", "ROUTE2", "FREQ2", "SUB3", "ROUTE3", "FREQ3", "ALCDRUG"])

COLUMN_HEADERS = ["gender_male", "gender_female", "gender_unknown",
                  "race_alaska", "race_native", "race_asian", "race_black", "race_white", "race_single", "race_multiple", "race_unknown",
                  "ethnic_puerto", "ethnic_mexican", "ethnic_cuban", "ethnic_none", "ethnic_latino", "ethnic_unknown",
                   "marstat_single", "marstat_married", "marstat_separated", "marstat_divorced", "marstat_unknown",
                  "employ_full", "employ_part", "employ_nlf", "employ_unemployed", "employ_unknown",
                  "pregnant", "preg_unknown",
                  "veteran", "vet_unknown",
                  "livarag_homeless", "livarag_dependent", "livarag_independent", "livarag_unknown",
                  "priminc_wages", "priminc_public", "priminc_retirement", "priminc_other", "priminc_none", "priminc_unknown",
                  "stfips_al", "stfips_ak", "stfips_az", "stfips_ar", "stfips_ca", "stfips_co", "stfips_ct",
                  "stfips_de", "stfips_dc", "stfips_fl", "stfips_hi", "stfips_id", "stfips_il", "stfips_in",
                  "stfips_ia", "stfips_ks", "stfips_ky", "stfips_la", "stfips_me", "stfips_md", "stfips_ma",
                  "stfips_mi", "stfips_mn", "stfips_ms", "stfips_mo", "stfips_mt", "stfips_ne", "stfips_nv",
                  "stfips_nh", "stfips_nj", "stfips_nm", "stfips_ny", "stfips_nc", "stfips_nd", "stfips_oh",
                  "stfips_ok", "stfips_pa", "stfips_ri", "stfips_sc", "stfips_sd", "stfips_tn", "stfips_tx",
                  "stfips_ut", "stfips_vt", "stfips_va", "stfips_wa", "stfips_wi", "stfips_wy", "stfips_pc",
                  "region_territory", "region_ne", "region_midwest", "region_south", "region_west",
                  "division_territory", "division_ne", "division_ma", "division_enc", "division_wnc",
                  "division_sa", "division_esc", "division_wsc", "division_mt", "division_pc",
                  "service_dt_24_h", "service_dt_24_r", "service_re_h", "service_re_short",
                  "service_re_long", "service_amb_in", "service_amb_nin", "service_amb_dt",
                  "methuse", "methuse_unknown",
                  "psource_individual", "psource_alc_drug", "psource_health", "psource_school",
                  "psource_employer", "psource_community", "psource_court", "psource_unknown",
                  "sub1_none", "sub1_alcohol", "sub1_cocaine", "sub1_weed", "sub1_heroin",
                  "sub1_methadone", "sub1_opiates", "sub1_pcp", "sub1_hallucinogens","sub1_meth",
                  "sub1_amphetamines", "sub1_stimulants", "sub1_benzos", "sub1_tranqs", "sub1_barbiturates",
                  "sub1_sedatives", "sub1_inhalants", "sub1_otc", "sub1_other", "sub1_unknown",
                  "route1_oral", "route1_smoke", "route1_inhalation", "route1_injection", "route1_other", "route1_unknown",
                  "freq1_none", "freq1_some", "freq1_daily", "freq1_unknown",
                  "sub2_none", "sub2_alcohol", "sub2_cocaine", "sub2_weed", "sub2_heroin",
                  "sub2_methadone", "sub2_opiates", "sub2_pcp", "sub2_hallucinogens","sub2_meth",
                  "sub2_amphetamines", "sub2_stimulants", "sub2_benzos", "sub2_tranqs", "sub2_barbiturates",
                  "sub2_sedatives", "sub2_inhalants", "sub2_otc", "sub2_other", "sub2_unknown",
                  "route2_oral", "route2_smoke", "route2_inhalation", "route2_injection", "route2_other", "route2_unknown",
                  "freq2_none", "freq2_some", "freq2_daily", "freq2_unknown",
                  "sub3_none", "sub3_alcohol", "sub3_cocaine", "sub3_weed", "sub3_heroin",
                  "sub3_methadone", "sub3_opiates", "sub3_pcp", "sub3_hallucinogens", "sub3_meth",
                  "sub3_amphetamines", "sub3_stimulants", "sub3_benzos", "sub3_tranqs", "sub3_barbiturates",
                  "sub3_sedatives", "sub3_inhalants", "sub3_otc", "sub3_other", "sub3_unknown",
                  "route3_oral", "route3_smoke", "route3_inhalation", "route3_injection", "route3_other", "route3_unknown",
                  "freq3_none", "freq3_some", "freq3_daily", "freq3_unknown",
                  "alcdrug_none", "alcdrug_alc", "alcdrug_drug", "alcdrug_both"]


one_hot_data = pd.DataFrame(np.zeros((data.shape[0], len(COLUMN_HEADERS) + reduced_data.shape[1])), columns=COLUMN_HEADERS + list(reduced_data.columns))

for i in range(data.shape[0]):
    to_fill = dict.fromkeys(COLUMN_HEADERS, 0)
    row = data.iloc[i]

    #gender
    temp = row["GENDER"]
    if temp == 1:
        to_fill["gender_male"] = 1
    elif temp == 2:
        to_fill["gender_female"] = 1
    elif temp == -9:
        to_fill["gender_unknown"] = 1

    #race
    temp = row["RACE"]
    if temp == 1:
        to_fill["race_alaska"] = 1
    elif temp == 2:
        to_fill["race_native"] = 1
    elif temp == 3 or temp == 6 or temp == 9:
        to_fill["race_asian"] = 1
    elif temp == 4:
        to_fill["race_black"] = 1
    elif temp == 5:
        to_fill["race_white"] = 1
    elif temp == 7:
        to_fill["race_single"] = 1
    elif temp == 8:
        to_fill["race_multiple"] = 1
    elif temp == -9:
        to_fill["race_unknown"] = 1

    #ethnicity
    temp = row["ETHNIC"]
    if temp == 1:
        to_fill["ethnic_puerto"] = 1
    elif temp == 2:
        to_fill["ethnic_mexican"] = 1
    elif temp == 3:
        to_fill["ethnic_cuban"] = 1
    elif temp == 4:
        to_fill["ethnic_none"] = 1
    elif temp == 5:
        to_fill["ethnic_latino"] = 1
    elif temp == -9:
        to_fill["ethnic_unknown"] = 1

    #martial status
    temp = row["MARSTAT"]
    if temp == 1:
        to_fill["marstat_single"] = 1
    elif temp == 2:
        to_fill["marstat_married"] = 1
    elif temp == 3:
        to_fill["marstat_separated"] = 1
    elif temp == 4:
        to_fill["marstat_divorced"] = 1
    elif temp == -9:
        to_fill["marstat_unknown"] = 1

    #employment status
    temp = row["EMPLOY"]
    if temp == 1:
        to_fill["employ_full"] = 1
    elif temp == 2:
        to_fill["employ_part"] = 1
    elif temp == 3:
        to_fill["employ_unemployed"] = 1
    elif temp == 4:
        to_fill["employ_nlf"] = 1
    elif temp == -9:
        to_fill["employ_unknown"] = 1

    #pregnant
    temp = row["PREG"]
    if temp == 1:
        to_fill["pregnant"] = 1
    elif temp == -9:
        to_fill["preg_unknown"] = 1

    #veteran
    temp = row["VET"]
    if temp == 1:
        to_fill["veteran"] = 1
    elif temp == -9:
        to_fill["vet_unknown"] = 1

    #living arrangements
    temp = row["LIVARAG"]
    if temp == 1:
        to_fill["livarag_homeless"] = 1
    elif temp == 2:
        to_fill["livarag_dependent"] = 1
    elif temp == 3:
        to_fill["livarag_independent"] = 1
    elif temp == -9:
        to_fill["livarag_unknown"] = 1

    #primary income
    temp = row["PRIMINC"]
    if temp == 1:
        to_fill["priminc_wages"] = 1
    elif temp == 2:
        to_fill["priminc_public"] = 1
    elif temp == 3:
        to_fill["priminc_retirement"] = 1
    elif temp == 4:
        to_fill["priminc_other"] = 1
    elif temp == 5:
        to_fill["priminc_none"] = 1
    elif temp == -9:
        to_fill["priminc_unknown"] = 1

    #number of arrests
    temp = row["ARRESTS"]
    if temp == 1:
        to_fill["priminc_wages"] = 1
    elif temp == 2:
        to_fill["priminc_public"] = 1
    elif temp == 3:
        to_fill["priminc_retirement"] = 1
    elif temp == -9:
        to_fill["priminc_unknown"] = 1

    #state
    temp = row["STFIPS"]
    if temp == 1:
        to_fill["stfips_al"] = 1
    elif temp == 2:
        to_fill["stfips_ak"] = 1
    elif temp == 4:
        to_fill["stfips_az"] = 1
    elif temp == 5:
        to_fill["stfips_ar"] = 1
    elif temp == 6:
        to_fill["stfips_ca"] = 1
    elif temp == 8:
        to_fill["stfips_co"] = 1
    elif temp == 9:
        to_fill["stfips_ct"] = 1
    elif temp == 10:
        to_fill["stfips_de"] = 1
    elif temp == 11:
        to_fill["stfips_dc"] = 1
    elif temp == 12:
        to_fill["stfips_fl"] = 1
    elif temp == 15:
        to_fill["stfips_hi"] = 1
    elif temp == 16:
        to_fill["stfips_id"] = 1
    elif temp == 17:
        to_fill["stfips_il"] = 1
    elif temp == 18:
        to_fill["stfips_in"] = 1
    elif temp == 19:
        to_fill["stfips_ia"] = 1
    elif temp == 20:
        to_fill["stfips_ks"] = 1
    elif temp == 21:
        to_fill["stfips_ky"] = 1
    elif temp == 22:
        to_fill["stfips_la"] = 1
    elif temp == 23:
        to_fill["stfips_me"] = 1
    elif temp == 24:
        to_fill["stfips_md"] = 1
    elif temp == 25:
        to_fill["stfips_ma"] = 1
    elif temp == 26:
        to_fill["stfips_mi"] = 1
    elif temp == 27:
        to_fill["stfips_mn"] = 1
    elif temp == 28:
        to_fill["stfips_ms"] = 1
    elif temp == 29:
        to_fill["stfips_mo"] = 1
    elif temp == 30:
        to_fill["stfips_mt"] = 1
    elif temp == 31:
        to_fill["stfips_ne"] = 1
    elif temp == 32:
        to_fill["stfips_nv"] = 1
    elif temp == 33:
        to_fill["stfips_nh"] = 1
    elif temp == 34:
        to_fill["stfips_nj"] = 1
    elif temp == 35:
        to_fill["stfips_nm"] = 1
    elif temp == 36:
        to_fill["stfips_ny"] = 1
    elif temp == 37:
        to_fill["stfips_nc"] = 1
    elif temp == 38:
        to_fill["stfips_nd"] = 1
    elif temp == 39:
        to_fill["stfips_oh"] = 1
    elif temp == 40:
        to_fill["stfips_ok"] = 1
    elif temp == 42:
        to_fill["stfips_pa"] = 1
    elif temp == 44:
        to_fill["stfips_ri"] = 1
    elif temp == 45:
        to_fill["stfips_sc"] = 1
    elif temp == 46:
        to_fill["stfips_sd"] = 1
    elif temp == 47:
        to_fill["stfips_tn"] = 1
    elif temp == 48:
        to_fill["stfips_tx"] = 1
    elif temp == 49:
        to_fill["stfips_ut"] = 1
    elif temp == 50:
        to_fill["stfips_vt"] = 1
    elif temp == 51:
        to_fill["stfips_va"] = 1
    elif temp == 53:
        to_fill["stfips_wa"] = 1
    elif temp == 55:
        to_fill["stfips_wi"] = 1
    elif temp == 56:
        to_fill["stfips_wy"] = 1
    elif temp == 72:
        to_fill["stfips_pc"] = 1

    #census region
    temp = row["REGION"]
    if temp == 0:
        to_fill["region_territory"] = 1
    elif temp == 1:
        to_fill["region_ne"] = 1
    elif temp == 2:
        to_fill["region_midwest"] = 1
    elif temp == 3:
        to_fill["region_south"] = 1
    elif temp == 4:
        to_fill["region_west"] = 1

    #census division
    temp = row["DIVISION"]
    if temp == 0:
        to_fill["division_territory"] = 1
    elif temp == 1:
        to_fill["division_ne"] = 1
    elif temp == 2:
        to_fill["division_ma"] = 1
    elif temp == 3:
        to_fill["division_enc"] = 1
    elif temp == 4:
        to_fill["division_wnc"] = 1
    elif temp == 5:
        to_fill["division_sa"] = 1
    elif temp == 6:
        to_fill["division_esc"] = 1
    elif temp == 7:
        to_fill["division_wsc"] = 1
    elif temp == 8:
        to_fill["division_mt"] = 1
    elif temp == 9:
        to_fill["division_pc"] = 1

    #service type
    temp = row["SERVICES"]
    if temp == 1:
        to_fill["service_dt_24_h"] = 1
    elif temp == 2:
        to_fill["service_dt_24_r"] = 1
    elif temp == 3:
        to_fill["service_re_h"] = 1
    elif temp == 4:
        to_fill["service_re_short"] = 1
    elif temp == 5:
        to_fill["service_re_long"] = 1
    elif temp == 6:
        to_fill["service_amb_in"] = 1
    elif temp == 7:
        to_fill["service_amb_nin"] = 1
    elif temp == 8:
        to_fill["service_amb_dt"] = 1

    #planned medication-assisted opioid therapy
    temp = row["METHUSE"]
    if temp == 1:
        to_fill["methuse"] = 1
    elif temp == -9:
        to_fill["methuse_unknown"] = 1

    #referral source
    temp = row["PSOURCE"]
    if temp == 1:
        to_fill["psource_individual"] = 1
    elif temp == 2:
        to_fill["psource_alc_drug"] = 1
    elif temp == 3:
        to_fill["psource_health"] = 1
    elif temp == 4:
        to_fill["psource_school"] = 1
    elif temp == 5:
        to_fill["psource_employer"] = 1
    elif temp == 6:
        to_fill["psource_community"] = 1
    elif temp == 7:
        to_fill["psource_court"] = 1
    elif temp == -9:
        to_fill["psource_unknown"] = 1

    #primary substance
    temp = row["SUB1"]
    if temp == 1:
        to_fill["sub1_none"] = 1
    elif temp == 2:
        to_fill["sub1_alcohol"] = 1
    elif temp == 3:
        to_fill["sub1_cocaine"] = 1
    elif temp == 4:
        to_fill["sub1_weed"] = 1
    elif temp == 5:
        to_fill["sub1_heroin"] = 1
    elif temp == 6:
        to_fill["sub1_methadone"] = 1
    elif temp == 7:
        to_fill["sub1_opiates"] = 1
    elif temp == 8:
        to_fill["sub1_pcp"] = 1
    elif temp == 9:
        to_fill["sub1_hallucinogens"] = 1
    elif temp == 10:
        to_fill["sub1_meth"] = 1
    elif temp == 11:
        to_fill["sub1_amphetamines"] = 1
    elif temp == 12:
        to_fill["sub1_stimulants"] = 1
    elif temp == 13:
        to_fill["sub1_benzos"] = 1
    elif temp == 14:
        to_fill["sub1_tranqs"] = 1
    elif temp == 15:
        to_fill["sub1_barbiturates"] = 1
    elif temp == 16:
        to_fill["sub1_sedatives"] = 1
    elif temp == 17:
        to_fill["sub1_inhalants"] = 1
    elif temp == 18:
        to_fill["sub1_otc"] = 1
    elif temp == 19:
        to_fill["sub1_other"] = 1
    elif temp == -9:
        to_fill["sub1_unknown"] = 1

    #route of administration for primary substance
    temp = row["ROUTE1"]
    if temp == 1:
        to_fill["route1_oral"] = 1
    elif temp == 2:
        to_fill["route1_smoke"] = 1
    elif temp == 3:
        to_fill["route1_inhalation"] = 1
    elif temp == 4:
        to_fill["route1_injection"] = 1
    elif temp == 5:
        to_fill["route1_other"] = 1
    elif temp == 6:
        to_fill["route1_unknown"] = 1

    #frequency of use for primary substance
    temp = row["FREQ1"]
    if temp == 1:
        to_fill["freq1_none"] = 1
    elif temp == 2:
        to_fill["freq1_some"] = 1
    elif temp == 3:
        to_fill["freq1_daily"] = 1
    elif temp == 6:
        to_fill["freq1_unknown"] = 1
    
    #secondary substance
    temp = row["SUB2"]
    if temp == 1:
        to_fill["sub2_none"] = 1
    elif temp == 2:
        to_fill["sub2_alcohol"] = 1
    elif temp == 3:
        to_fill["sub2_cocaine"] = 1
    elif temp == 4:
        to_fill["sub2_weed"] = 1
    elif temp == 5:
        to_fill["sub2_heroin"] = 1
    elif temp == 6:
        to_fill["sub2_methadone"] = 1
    elif temp == 7:
        to_fill["sub2_opiates"] = 1
    elif temp == 8:
        to_fill["sub2_pcp"] = 1
    elif temp == 9:
        to_fill["sub2_hallucinogens"] = 1
    elif temp == 10:
        to_fill["sub2_meth"] = 1
    elif temp == 11:
        to_fill["sub2_amphetamines"] = 1
    elif temp == 12:
        to_fill["sub2_stimulants"] = 1
    elif temp == 13:
        to_fill["sub2_benzos"] = 1
    elif temp == 14:
        to_fill["sub2_tranqs"] = 1
    elif temp == 15:
        to_fill["sub2_barbiturates"] = 1
    elif temp == 16:
        to_fill["sub2_sedatives"] = 1
    elif temp == 17:
        to_fill["sub2_inhalants"] = 1
    elif temp == 18:
        to_fill["sub2_otc"] = 1
    elif temp == 19:
        to_fill["sub2_other"] = 1
    elif temp == -9:
        to_fill["sub2_unknown"] = 1

    #route of administration for secondary substance
    temp = row["ROUTE2"]
    if temp == 1:
        to_fill["route2_oral"] = 1
    elif temp == 2:
        to_fill["route2_smoke"] = 1
    elif temp == 3:
        to_fill["route2_inhalation"] = 1
    elif temp == 4:
        to_fill["route2_injection"] = 1
    elif temp == 5:
        to_fill["route2_other"] = 1
    elif temp == 6:
        to_fill["route2_unknown"] = 1

    #frequency of use for secondary substance
    temp = row["FREQ2"]
    if temp == 1:
        to_fill["freq2_none"] = 1
    elif temp == 2:
        to_fill["freq2_some"] = 1
    elif temp == 3:
        to_fill["freq2_daily"] = 1
    elif temp == 6:
        to_fill["freq2_unknown"] = 1

    #tertiary substance
    temp = row["SUB3"]
    if temp == 1:
        to_fill["sub3_none"] = 1
    elif temp == 2:
        to_fill["sub3_alcohol"] = 1
    elif temp == 3:
        to_fill["sub3_cocaine"] = 1
    elif temp == 4:
        to_fill["sub3_weed"] = 1
    elif temp == 5:
        to_fill["sub3_heroin"] = 1
    elif temp == 6:
        to_fill["sub3_methadone"] = 1
    elif temp == 7:
        to_fill["sub3_opiates"] = 1
    elif temp == 8:
        to_fill["sub3_pcp"] = 1
    elif temp == 9:
        to_fill["sub3_hallucinogens"] = 1
    elif temp == 10:
        to_fill["sub3_meth"] = 1
    elif temp == 11:
        to_fill["sub3_amphetamines"] = 1
    elif temp == 12:
        to_fill["sub3_stimulants"] = 1
    elif temp == 13:
        to_fill["sub3_benzos"] = 1
    elif temp == 14:
        to_fill["sub3_tranqs"] = 1
    elif temp == 15:
        to_fill["sub3_barbiturates"] = 1
    elif temp == 16:
        to_fill["sub3_sedatives"] = 1
    elif temp == 17:
        to_fill["sub3_inhalants"] = 1
    elif temp == 18:
        to_fill["sub3_otc"] = 1
    elif temp == 19:
        to_fill["sub3_other"] = 1
    elif temp == -9:
        to_fill["sub3_unknown"] = 1

    #route of administration for tertiary substance
    temp = row["ROUTE3"]
    if temp == 1:
        to_fill["route3_oral"] = 1
    elif temp == 2:
        to_fill["route3_smoke"] = 1
    elif temp == 3:
        to_fill["route3_inhalation"] = 1
    elif temp == 4:
        to_fill["route3_injection"] = 1
    elif temp == 5:
        to_fill["route3_other"] = 1
    elif temp == 6:
        to_fill["route3_unknown"] = 1

    #frequency of use for tertiary substance
    temp = row["FREQ3"]
    if temp == 1:
        to_fill["freq3_none"] = 1
    elif temp == 2:
        to_fill["freq3_some"] = 1
    elif temp == 3:
        to_fill["freq3_daily"] = 1
    elif temp == 6:
        to_fill["freq3_unknown"] = 1

    #category of substance use
        # frequency of use for tertiary substance
        temp = row["ALCDRUG"]
        if temp == 0:
            to_fill["alcdrug_none"] = 1
        elif temp == 1:
            to_fill["alcdrug_alc"] = 1
        elif temp == 2:
            to_fill["alcdrug_drug"] = 1
        elif temp == 3:
            to_fill["alcdrug_both"] = 1

    to_fill.update(reduced_data.iloc[[i]].to_dict(orient="records")[0])
    one_hot_data.iloc[i] = to_fill
    if i % 1000 == 0:
        print(f"Iteration: {i}")

f = open("500000 teds/one_hot_2017_1.csv", "x")
one_hot_data.to_csv(f, index=False)