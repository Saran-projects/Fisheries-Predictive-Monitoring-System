import pandas as pd
import os

def main():
    print("Loading datasets...")
    base_dir = "GlobalProduction_2026.1.0"
    
    # Read main production data
    # Some columns might have mixed types, so let's load it carefully
    # Also drop NaN values in 'VALUE' as we can't train on them
    prod_df = pd.read_csv(f"{base_dir}/Global_production_quantity.csv")
    prod_df = prod_df.dropna(subset=['VALUE'])
    
    # Read mapping tables
    countries_df = pd.read_csv(f"{base_dir}/CL_FI_COUNTRY_GROUPS.csv", dtype={'UN_Code': str})
    species_df = pd.read_csv(f"{base_dir}/CL_FI_SPECIES_GROUPS.csv")
    areas_df = pd.read_csv(f"{base_dir}/CL_FI_WATERAREA_GROUPS.csv", dtype={'Code': str})
    
    # The production dataset columns:
    # "COUNTRY.UN_CODE","SPECIES.ALPHA_3_CODE","AREA.CODE","PRODUCTION_SOURCE_DET.CODE","MEASURE","PERIOD","VALUE","STATUS"
    
    # Ensure types match for merging
    prod_df['COUNTRY.UN_CODE'] = prod_df['COUNTRY.UN_CODE'].astype(str).str.zfill(3)
    countries_df['UN_Code'] = countries_df['UN_Code'].astype(str).str.zfill(3)
    
    prod_df['AREA.CODE'] = prod_df['AREA.CODE'].astype(str).str.zfill(2)
    areas_df['Code'] = areas_df['Code'].astype(str).str.zfill(2)
    
    print("Merging Country mapping...")
    # Merge countries
    merged = pd.merge(prod_df, countries_df[['UN_Code', 'ISO3_Code', 'Name_En']], 
                      left_on='COUNTRY.UN_CODE', right_on='UN_Code', how='left')
    merged.rename(columns={'Name_En': 'Country', 'ISO3_Code': 'ISO3'}, inplace=True)
    
    print("Merging Species mapping...")
    # Merge species
    merged = pd.merge(merged, species_df[['3A_Code', 'Name_En', 'Major_Group']], 
                      left_on='SPECIES.ALPHA_3_CODE', right_on='3A_Code', how='left')
    merged.rename(columns={'Name_En': 'Species'}, inplace=True)
    
    print("Merging Area mapping...")
    # Merge areas
    merged = pd.merge(merged, areas_df[['Code', 'Name_En']], 
                      left_on='AREA.CODE', right_on='Code', how='left')
    merged.rename(columns={'Name_En': 'Area'}, inplace=True)
    
    # Filter to relevant columns and clean up
    final_cols = ['PERIOD', 'Country', 'ISO3', 'Species', 'Major_Group', 'Area', 'MEASURE', 'VALUE']
    final_df = merged[final_cols].copy()
    
    # Rename for consistency
    final_df.rename(columns={'PERIOD': 'Year', 'VALUE': 'Production_Quantity'}, inplace=True)
    
    # Drop rows where critical fields are missing
    final_df = final_df.dropna(subset=['Country', 'Species', 'Year', 'Production_Quantity'])
    
    # Sort by Year
    final_df = final_df.sort_values('Year')
    
    # Filter recent years to keep dataset size manageable for app performance (e.g. 2000 onwards)
    # The dataset has data from 1950, which makes the file very large
    final_df = final_df[final_df['Year'] >= 2000]
    
    # Aggregate values in case there are duplicates (e.g., multiple production sources)
    agg_df = final_df.groupby(['Year', 'Country', 'ISO3', 'Species', 'Major_Group', 'Area'])['Production_Quantity'].sum().reset_index()
    
    print("Exporting to global_data.csv...")
    agg_df.to_csv("global_data.csv", index=False)
    print(f"Done! Processed dataset has {len(agg_df)} rows.")

if __name__ == "__main__":
    main()
