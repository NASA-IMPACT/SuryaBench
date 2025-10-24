import os
import re
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from sunpy.net import Fido, attrs as a
from typing import Optional

'''
This dataset creation code is based on the repository: https://bitbucket.org/gsudmlab/flare_list_creator/src/main/.  
Please contact the following if you have any issues or questions:  
Jinsu Hong: jhong36@gsu.edu  
Berkay Aydin: baydin2@gsu.edu
'''

# -------------------------------------------------------------------
# --- 1. Query Flare Lists ------------------------------------------
# -------------------------------------------------------------------


def query_goes_flarelist(
        tstart: str = "2010/05/01",
        tend: str = "2025/01/01",
        file_path: str = "goes_flarelist.csv"
        ) -> pd.DataFrame:
    """Query GOES flare events from HEK and save to CSV."""
    event_type = "FL"
    print(f"Querying GOES flare list from {tstart} to {tend}...")
    result = Fido.search(
        a.Time(tstart, tend),
        a.hek.EventType(event_type),
        a.hek.OBS.Observatory == "GOES"

    )
    hek_results = result["hek"]
    filtered = hek_results[
        "event_starttime", "event_peaktime", "event_endtime",
        "fl_goescls", "hgs_x", "hgs_y",
        "hgc_x", "hgc_y", "ar_noaanum"
    ]
    filtered.write(file_path, format="csv", overwrite=True)
    df = pd.read_csv(file_path)
    df.rename(columns={
        "hgs_x": "hgs_lon",
        "hgs_y": "hgs_lat",
        "hgc_x": "hgc_lon",
        "hgc_y": "hgc_lat"
    }, inplace=True)
    df.to_csv(file_path, index=False)
    print(f"Saved GOES flare catalog to {file_path}")
    return df


def query_aia_flarelist(
        tstart: str = "2010/05/01",
        tend: str = "2025/01/01",
        file_path: str = "aia_flarelist.csv"
        ) -> pd.DataFrame:
    """Query AIA flare events from HEK and save to CSV."""
    event_type = "FL"
    print(f"Querying AIA flare list from {tstart} to {tend}...")
    result = Fido.search(
        a.Time(tstart, tend),
        a.hek.EventType(event_type),
        a.hek.OBS.Observatory == "SDO"
    )
    hek_results = result["hek"]
    filtered = hek_results[
        "event_starttime", "event_peaktime", "event_endtime",
        "fl_goescls", "obs_channelid", "obs_observatory",
        "search_frm_name", "ar_noaanum",
        "hgs_x", "hgs_y", "hgc_x", "hgc_y"
    ]
    filtered.write(file_path, format="csv", overwrite=True)
    df = pd.read_csv(file_path)
    df.rename(columns={
        "hgs_x": "hgs_aia_lon",
        "hgs_y": "hgs_aia_lat",
        "hgc_x": "hgc_aia_lon",
        "hgc_y": "hgc_aia_lat"
    }, inplace=True)
    df.to_csv(file_path)
    print(f"Saved AIA flare catalog to {file_path}")
    return pd.read_csv(file_path)

# -------------------------------------------------------------------
# --- 2. SSW Location Scraper ---------------------------------------
# -------------------------------------------------------------------


def scrape_ssw_location(
        file_path: str,
        base_url: str = "https://www.lmsal.com/solarsoft/latest_events_archive/events_summary/"
        ) -> pd.DataFrame:
    """Scrape SSW location and NOAA AR info for each flare entry."""
    df = pd.read_csv(file_path)
    df['event_starttime'] = pd.to_datetime(df['event_starttime'])

    position_records = []

    print("Scraping SSW flare locations...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scraping SSW"):
        dt = row['event_starttime']
        url = f"{base_url}{dt.year}/{dt.month:02d}/{dt.day:02d}/gev_{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}/index.html"

        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                position_records.append([None, None, None])
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            tables = soup.find_all("table")

            candidate_id = f"gev_{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}"
            found_entry = [None, None, None]

            for table in tables:
                cells = re.findall("<td>(.*?)</td>", str(table))
                if len(cells) > 6 and cells[1] == candidate_id:
                    position_html = cells[6]
                    if ">" in position_html:
                        noaa_ar = position_html.split("=")[5][:5]
                        position = position_html.split(">")[2][1:7]
                        found_entry = [cells[1], position, noaa_ar]
                    else:
                        found_entry = [cells[1], position_html, None]
                    break

            position_records.append(found_entry)
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            position_records.append([None, None, None])

    df[["ssw_id", "location_ssw", "noaa_ar_ssw"]] = pd.DataFrame(position_records)
    df.to_csv(file_path, index=False)
    print(f"Updated {file_path} with SSW position info.")
    return df

# -------------------------------------------------------------------
# --- 3. Convert SSW Location to Numeric -----------------------------
# -------------------------------------------------------------------


def parse_ssw_location(file_path: str) -> pd.DataFrame:
    """Parse SSW location strings into numeric lat/lon."""
    df = pd.read_csv(file_path)

    def parse_coord(loc: Optional[str]):
        if pd.isna(loc) or len(loc) < 6:
            return None, None
        try:
            lon, lat = loc[:3], loc[3:6]
            lon_val = int(lon[1:]) * (1 if "N" in lon else -1)
            lat_val = int(lat[1:]) * (-1 if "E" in lat else 1)
            return lat_val, lon_val
        except Exception:
            return None, None

    coords = df["location_ssw"].apply(parse_coord)
    df["lat_ssw"], df["lon_ssw"] = zip(*coords)
    df.to_csv(file_path, index=False)
    return df

# -------------------------------------------------------------------
# --- 4. Compare NOAA vs SSW ----------------------------------------
# -------------------------------------------------------------------


def compare_noaa_ssw(file_path: str) -> pd.DataFrame:
    """Compare NOAA vs SSW AR and coordinates."""
    df = pd.read_csv(file_path)

    df["AR_valid"] = np.where(
        (df["ar_noaanum"] != df["noaa_ar_ssw"]) &
        (df["ar_noaanum"] != 0) &
        (df["noaa_ar_ssw"].notnull()),
        False, True
    )

    df["dist_noaa_ssw"] = df.apply(
        lambda x: ((x["hgs_lat"] - x["lat_ssw"]) ** 2 + (x["hgs_lon"] - x["lon_ssw"]) ** 2) ** 0.5
        if pd.notnull(x["lat_ssw"]) and pd.notnull(x["lon_ssw"]) else None,
        axis=1
    ).round(1)

    df.to_csv(file_path, index=False)
    print("NOAA–SSW comparison complete.")
    return df

# -------------------------------------------------------------------
# --- 5. Combine GOES and AIA ---------------------------------------
# -------------------------------------------------------------------


def combine_goes_aia(goes_path: str, aia_path: str) -> pd.DataFrame:
    """Combine GOES and AIA flare lists based on time proximity (±3 min)."""
    df_goes = pd.read_csv(goes_path)
    df_aia = pd.read_csv(aia_path)
    df_aia = df_aia.loc[df_aia['obs_channelid'] != 'EUV']

    for col in ["event_starttime", "event_peaktime", "event_endtime"]:
        df_goes[col] = pd.to_datetime(df_goes[col])
        df_aia[col] = pd.to_datetime(df_aia[col])

    print("Matching GOES and AIA flares by timestamps...")
    result_rows = []
    for _, row in tqdm(df_goes.iterrows(), total=len(df_goes), desc="Matching"):
        start, peak, end = row["event_starttime"], row["event_peaktime"], row["event_endtime"]
        mask = (
            df_aia["event_starttime"].between(start - pd.Timedelta(minutes=3), start + pd.Timedelta(minutes=3)) &
            df_aia["event_peaktime"].between(peak - pd.Timedelta(minutes=3), peak + pd.Timedelta(minutes=3)) &
            df_aia["event_endtime"].between(end - pd.Timedelta(minutes=3), end + pd.Timedelta(minutes=3))
        )
        matched = df_aia[mask].head(1)
        result_rows.append(matched.values.tolist()[0] if not matched.empty else [None] * len(df_aia.columns))

    df_match = pd.DataFrame(result_rows, columns=df_aia.columns)
    df_match = df_match[[
        "obs_channelid", "obs_observatory",
        "search_frm_name", "ar_noaanum",
        "hgs_aia_lon", "hgs_aia_lat", "hgc_aia_lon", "hgc_aia_lat"
        ]]
    df_combined = pd.concat([df_goes, df_match], axis=1)
    df_combined.to_csv(goes_path, index=False)
    print(f"Combined catalog saved to {goes_path}")
    return df_combined

# -------------------------------------------------------------------
# --- 6. Main Execution ----------------------------------------------
# -------------------------------------------------------------------


if __name__ == "__main__":
    start = "2010/05/01"
    end = "2010/08/01"
    goes_flare_file_path = "flare_catalog.csv"
    aia_flare_file_path = "aia_flare_catalog.csv"

    print("Creating GOES and AIA flare catalogs...")
    goes_df = query_goes_flarelist(start, end, goes_flare_file_path)
    aia_df = query_aia_flarelist(start, end, aia_flare_file_path)

    print("Scraping and validating SSW locations...")
    goes_df = scrape_ssw_location(goes_flare_file_path)
    goes_df = parse_ssw_location(goes_flare_file_path)
    goes_df = compare_noaa_ssw(goes_flare_file_path)

    print("Merging AIA with GOES data...")
    combine_goes_aia(goes_flare_file_path, aia_flare_file_path)
