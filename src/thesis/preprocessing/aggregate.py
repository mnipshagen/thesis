from functools import partial

import numpy as np
import pandas as pd

def _aggregate_sibling(g, data_dict):
    """
    Aggregate sibling file by determining:
    - highest degree of most siblings
    - how many different degrees siblings have
    - if the student is youngest, oldest or inbetween
    """
    idt = g["ID_t"].iloc[0]
    birth = data_dict["pTargetCATI"][data_dict["pTargetCATI"]["ID_t"] == idt][["t70000m", "t70000y"]]
    birthdate = np.datetime64(
        birth["t70000y"].astype(int).astype(str).iloc[0] + "-" +
        f'{birth["t70000m"].astype(int).astype(str).iloc[0]:0>2}-01'
    )
    # Sibling’s highest school-leaving qualification
    most_leaving_cert = _aggregate_get_item(g["tg32711"], "mode")
    amount_leaving_cert = len(g["tg32711"].unique())
    g["birthdate"] = g["tg3270y"].astype(int).astype(
        str) + "-" + g["tg3270m"].astype(int).astype(str) + "-01"
    g["birthdate"] = g["birthdate"].astype("datetime64")
    oldest = all(g["birthdate"] < birthdate)
    youngest = all(g["birthdate"] > birthdate)
    return pd.DataFrame({
        # "ID_t": idt,
        "sibling_leaving_cert": most_leaving_cert,
        "sibling_amount_leaving_certs": amount_leaving_cert,
        "age_position": 0 if youngest else (2 if oldest else 1)
    }, index=[g["ID_t"].iloc[0]])


def _aggregate_military(g):
    """
    Aggregate military spell file by determining:
    - median end month of military episodes
    - median start month of episodes
    - median type of episode
    - whether student participated in seminars during most of episodes
    """
    end_month = _aggregate_get_item(g["ts2112m_g1"], "median") # end month
    start_month = _aggregate_get_item(g["ts2111m_g1"], "median") # start month
    type_episode = _aggregate_get_item(g["ts21201"], "median") # type of episode
    attendance = _aggregate_get_item(g["ts21202"], "median") # attendance of seminars during service
    return pd.DataFrame({
        # "ID_t": g["ID_t"].iloc[0],
        "military_end_month": end_month,
        "military_start_month": start_month,
        "military_episode_type": type_episode,
        "military_seminar_attendance": attendance
    }, index=[g["ID_t"].iloc[0]])


def _aggregate_gap(g):
    """
    Aggregate gap spell file by determining:
    - median end month of gap
    - median start month of gap
    - median type of gap
    """
    end_month = _aggregate_get_item(g["ts2912m_g1"], "median")  # end month
    start_month = _aggregate_get_item(g["ts2911m_g1"], "median")  # start month
    type_episode = _aggregate_get_item(g["ts29101"], "median")  # type of episode
    return pd.DataFrame({
        # "ID_t": g["ID_t"].iloc[0],
        "gap_end_month": end_month,
        "gap_start_month": start_month,
        "gap_episode_type": type_episode,
    }, index=[g["ID_t"].iloc[0]])


def _aggregate_unemp(g):
    """
    Aggregate unemployment spell file by determining:
    - median start month
    - median end month
    - avg number of applications rounded to nearest integer
    - avg number of interviews rounded to nearest integer
    """
    start_month = _aggregate_get_item(g["ts2511m_g1"], "median")  # start month
    end_month = _aggregate_get_item(g["ts2512m_g1"], "median")  # end month
    no_applications = _aggregate_get_item(g["ts25205"], "mean") # no of applications
    no_interviews = _aggregate_get_item(g["ts25207"], "mean") # no of interviews
    return pd.DataFrame({
        # "ID_t": g["ID_t"].iloc[0],
        "unemployment_end_month": end_month,
        "unemployment_start_month": start_month,
        "unemployment_number_of_applications": no_applications,
        "unemployment_number_of_interviews": no_interviews
    }, index=[g["ID_t"].iloc[0]])


def _aggregate_internship(g):
    """
    Aggregate intenrship spell file by determining:
    - median start month
    - median end month
    - avg amount of remuneration rounded to nearest integer
    - avg working time rounded to nearest integer
    """
    start_month = _aggregate_get_item(g["tg3607m_g1"], "median")  # start month
    end_month = _aggregate_get_item(g["tg3608m_g1"], "median")  # end month
    remuneration = _aggregate_get_item(g["tg36114"], "mean") # amount of remuneration
    avg_working_time = _aggregate_get_item(g["tg36111"], "mean") # avg working time
    return pd.DataFrame({
        # "ID_t": g["ID_t"].iloc[0],
        "internship_end_month": end_month,
        "internship_start_month": start_month,
        "internship_remuneration_amount": remuneration,
        "internship_average_working_time": avg_working_time
    }, index=[g["ID_t"].iloc[0]])


def _aggregate_school(g):
    """
    Aggregate school spell file by determining:
    - overall grade of most recent school certificate
    - the 4 first "abitur" subjects taken for their leaving cert <-- this might be nan
    - start year of most recent school certificate
    - end year of most recent school certificate
    - points in math course in most recent school certificate <-- this might be nan
    - points in german course in most recent school certificate <-- this might be nan
    """
    grade_last_cert = _aggregate_get_item(g["ts11218"], "latest") # most recent grade (most likely leaving cert)
    subject_1 = _aggregate_get_item(g["t724801"], "latest") # 1st abitur subject, might be nan for non-abitur cert
    subject_2 = _aggregate_get_item(g["t724802"], "latest") # 2nd, s.o.
    subject_3 = _aggregate_get_item(g["t724803"], "latest") # 3rd, s.o.
    subject_4 = _aggregate_get_item(g["t724804"], "latest") # 4th. s.o.
    start_year = _aggregate_get_item(g["ts1111y_g1"], "latest") # start year of school year
    end_year = _aggregate_get_item(g["ts1112y_g1"], "latest") # end year of school year
    math_points = _aggregate_get_item(g["t724712"], "latest") # math points in leaving cert
    german_points = _aggregate_get_item(g["t724714"], "latest") # german points in leaving cert

    return pd.DataFrame({
        # "ID_t": g["ID_t"].iloc[0],
        "school_last_grade" : grade_last_cert,
        "school_subject_1" : subject_1,
        "school_subject_2" : subject_2,
        "school_subject_3" : subject_3,
        "school_subject_4" : subject_4,
        "school_start_year" : start_year,
        "school_end_year" : end_year,
        "school_math_points" : math_points,
        "school_german_points" : german_points
    }, index=[g["ID_t"].iloc[0]])


def _aggregate_partner(g):
    start_m = _aggregate_get_item(g["tg2811m"], "median")  # Start date Partnership Month
    end_m = _aggregate_get_item(g["tg2804m"], "median")  # End date Partnership episode (month)
    # g["ts31226_g2"]  # partner: occupation (KldB 2010)
    # g["ts31212_g1"]  # Partner: Highest educational achievement (ISCED-97)
    avg_work_hours = _aggregate_get_item(g["ts31224_v1"], "mean") # Working hours, partner.
    contact_freq = _aggregate_get_item(g["t733005"], "mean") # Frequency of contact with partner
    return pd.DataFrame({
        # "ID_t": g["ID_t"].iloc[0],
        "partner_start_month" : start_m,
        "partner_end_month" : end_m,
        "partner_avg_working_hours": avg_work_hours,
        "partner_contact_frequency" : contact_freq
    }, index=[g["ID_t"].iloc[0]])


def _aggregate_employment(g):
    study_relation = _aggregate_get_item(g["tg23228"], "median")  # Relation to studies of non-student employment
    descr = _aggregate_get_item(g["ts23201_g2"], "mode") # Job description (KldB 2010)
    # Type of education required
    edu_required = _aggregate_get_item(g["ts23228"], "mode")
    working_hours = _aggregate_get_item(g["ts23223"], "mean") # Actual weekly working hours at the moment/at the end
    gross_income = _aggregate_get_item(g["ts23510"], "mean")  # Gross income, open
    net_income = _aggregate_get_item(g["ts23410"], "mean")  # Net income, open
    return pd.DataFrame({
        # "ID_t": g["ID_t"].iloc[0],
        "employment_relation_to_studies": study_relation,
        "employment_job_description": descr,
        "employment_required_education": edu_required,
        "employment_average_working_hours": working_hours,
        "employment_average_gross_income": gross_income,
        "employment_average_net_income": net_income
    }, index=[g["ID_t"].iloc[0]])


def _aggregate_get_item(var, agg_type):
    agg_func = {
        "mode" : lambda x: x.mode().iloc[0] if x.mode().shape[0] else np.nan,
        "mean" : lambda x: np.round(x.mean() + 0.5),
        "median" : lambda x: x.median(),
        "latest" : lambda x: x.iloc[-1]
    }
    # var = g[var_name]
    return agg_func[agg_type](var)


def _aggregate_spell_files(valid_ids, data_dict):
    """
    Since we need to put all stuff into 1 multidimensional vector, we need to aggregate the data from the spell files so that we are left with 1 row for each ID.
    """
    if _verbose:
        print("Starting aggregation...")
    # the var names of start and end dates for all relevant dataframes
    datevars = {
        "spGap": {"start_m": "ts2911m_g1", "start_y": "ts2911y_g1", "end_m": "ts2912m_g1", "end_y": "ts2912y_g1"},
        "spMilitary": {"start_m": "ts2111m_g1", "start_y": "ts2111y_g1", "end_m": "ts2112m_g1", "end_y": "ts2112y_g1"},
        "spUnemp": {"start_m": "ts2511m_g1", "start_y": "ts2511y_g1", "end_m": "ts2512m_g1", "end_y": "ts2512y_g1"},
        "spIntenrship": {"start_m": "tg3607m_g1", "start_y": "tg3607y_g1", "end_m": "tg3608m_g1", "end_y": "tg3608y_g1"},
        "spPartner": {"start_m": "tg2811m", "start_y": "tg2811y", "end_m": "tg2804m", "end_y": "tg2804y"},
        "spEmp": {"start_m": "ts2311m_g1", "start_y": "ts2311y_g1", "end_m": "ts2312m_g1", "end_y": "ts2312y_g1"},
        "spInternship": {"start_m": "tg3607m_g1", "start_y": "tg3607y_g1", "end_m": "tg3608m_g1", "end_y": "tg3608m_g1"},
        "pTargetCATI": {"start_m": "start_m", "start_y": "start_y", "end_m": None, "end_y": None},
        "pTargetCAWI": {"start_m": "start_m", "start_y": "start_y", "end_m": None, "end_y": None}
    }
    # map wave no to year
    wave2year = {
        1:2011,
        2:2011,
        3:2012,
        4:2012,
        5:2013,
        6:2013,
        7:2014,
        8:2014,
        9:2015,
        10:2016,
        11:2016
    }
    basics = pd.read_stata("./data/raw/SC5_Basics_D_11-0-0.dta", convert_categoricals=False)
    data_dict["pTargetCATI"].drop(columns=["t70000m", "t70000y"], inplace=True)
    data_dict["pTargetCATI"] = pd.merge(data_dict["pTargetCATI"], basics[["ID_t", "t70000m", "t70000y"]], on="ID_t", how="inner")
    for key in ["pTargetCAWI", "pTargetCATI"]:
        data_dict[key]["start_m"] = 1
        data_dict[key]["start_y"] = data_dict[key]["wave"].apply(lambda x: wave2year[x])
    # get only valid voctrain spells
    df = pd.merge(valid_ids, data_dict["spVocTrain"], on=["ID_t", "spell"], how="inner")
    # get end date of last episode for each id. no filtering for each single spell. too complex.
    latest_date_per_id = df.groupby("ID_t").apply(lambda x: x.sort_values("spell").iloc[-1][["ID_t", "ts1512m_g1", "ts1512y_g1"]]).reset_index(drop=True)

    if _verbose:
        print("Done preperation: added columns to CATI/CAWI and created latest date dataframe.")

    # create a new dictionary which holds only the spells which have started before the last voctrain spell ended
    time_limited_data_dict = {}
    for dfk in data_dict.keys():
        if dfk not in datevars.keys():
            if _verbose:
                print(f"Skipped {dfk}")
            continue
        if _verbose:
            print(f"Filtering {dfk}")
        df = data_dict[dfk]
        # join with voctrain end date
        df = pd.merge(df, latest_date_per_id, on="ID_t")
        # get var names of start dates
        start_m = datevars[dfk]["start_m"]
        start_y = datevars[dfk]["start_y"]

        def f(g):
            """
            Filter spells out that start after the date stored for this id
            """
            out = None
            #print(f"Doing {dfk}: Date {g['ts1512m_g1'].iloc[0]}.{g['ts1512y_g1'].iloc[0]}")
            for row in g.iterrows():
                d = row[1]
                # if the current spell starts later then enddate...
                if d[start_y] > d["ts1512y_g1"] or (d[start_y] == d["ts1512y_g1"] and d[start_m] > d["ts1512m_g1"]):
                    #print(f"Left out row with start date {d[start_m]}.{d[start_y]}")
                    continue
                elif out is None:
                    out = pd.DataFrame(d).T
                else: # concatenate all valid ros
                    #print("Included one..")
                    out = pd.concat([out, pd.DataFrame(d).T])
                    #out.append(row[0])
            # making sure there is **some** data left. this is for the 164 people who have only valid spells that end in 2009 and earlier
            if out is None:
                out = pd.DataFrame(g.iloc[0]).T
            # return cleaned df group
            return out
        
        # save into a new dict. reset_index to drop the unnecessary ID_t index column
        time_limited_data_dict[dfk] = df.groupby("ID_t").apply(f).reset_index(drop=True)

    if _verbose:
        print("Created Time Limited Data Dict.")

    def apply_agg(df, func):
        return df.groupby("ID_t").apply(func).reset_index(level=1, drop=True).reset_index()

    def agg_sibling(sib):
        mask = sib[["tg3270m", "tg3270y"]] >= 0
        sib[["tg3270m", "tg3270y"]] = sib[["tg3270m", "tg3270y"]].where(mask, other=np.nan)
        sib.dropna(subset=["tg3270m", "tg3270y"], inplace=True)
        sib["tg3270m"].loc[sib["tg3270m"] == 21] = 1
        sib["tg3270m"].loc[sib["tg3270m"] == 24] = 4
        sib["tg3270m"].loc[sib["tg3270m"] == 27] = 7
        sib["tg3270m"].loc[sib["tg3270m"] == 30] = 10
        sib["tg3270m"].loc[sib["tg3270m"] == 32] = 12
        return apply_agg(sib, partial(_aggregate_sibling, data_dict=data_dict))

    def agg_military(mil):
        return apply_agg(mil, _aggregate_military)

    def agg_gap(gap):
        return apply_agg(gap, _aggregate_gap)

    def agg_unemp(unemp):
        return apply_agg(unemp, _aggregate_unemp)

    def agg_internship(intern):
        return apply_agg(intern, _aggregate_internship)

    def agg_school(school):
        return apply_agg(school, _aggregate_school)

    def agg_partner(partner):
        return apply_agg(partner, _aggregate_partner)

    def agg_emp(emp):
        return apply_agg(emp, _aggregate_employment)

    def agg_cawi(cawi):
        return cawi.sort_values(["ID_t", "wave"]).drop_duplicates(subset="ID_t")
    
    def agg_cati(cati):
        return cati.sort_values(["ID_t", "wave"]).drop_duplicates(subset="ID_t")

    agg_dict = {
        "spSibling": agg_sibling,
        "spMilitary": agg_military,
        "spGap": agg_gap,
        "spUnemp": agg_unemp,
        "spInternship": agg_internship,
        "spSchool": agg_school,
        "spPartner": agg_partner,
        "spEmp": agg_emp,
        "pTargetCAWI": agg_cawi,
        "pTargetCATI": agg_cati
    }
    agg_data_dict = {}
    for df_k in agg_dict.keys():
        if _verbose:
            print(f"Aggregating {df_k}...", end=" ")
        if df_k in time_limited_data_dict.keys():
            df = time_limited_data_dict[df_k]
        else:
            df = data_dict[df_k]
        agg_data_dict[df_k] = agg_dict[df_k](df)
        if _verbose:
            print("Done.")
    
    if _verbose:
        print("Aggregating done.")
        
    return agg_data_dict


def agg(valid_ids, data_dict, **kwargs):
    """
    Main function which starts the aggregation on all files.

    If supplied, will use multithreading capacities.
    """
    for k in kwargs:
        if k == "verbose":
            global _verbose
            _verbose = kwargs[k]
        elif k == "mp_pool":
            global _pool
            _pool = kwargs[k]
        elif k == "mp_manager":
            global _manager
            _manager = kwargs[k]
        else:
            print(f"Did not recognise {k} as a valid parameter.")
    return _aggregate_spell_files(valid_ids, data_dict)