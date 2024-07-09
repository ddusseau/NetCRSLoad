import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob2
import scipy.stats as st
import seaborn as sns
import cpi

## function to read large CSV file
def read_csv(file_name, fields):
    for chunk in pd.read_csv(file_name, chunksize=1000000, usecols=fields, engine="c"):
        yield chunk

## preprocess NFIP data for date and fields of interest
def preprocess(doi):
    fields = ['propertyState', 'countyCode', 'policyEffectiveDate','crsClassCode','policyTerminationDate', 'occupancyType','nfipRatedCommunityNumber','nfipCommunityNumberCurrent','totalInsurancePremiumOfThePolicy','rateMethod','communityProbationSurcharge','reserveFundAssessment','federalPolicyFee']
    pd_dataframes = []
    for df in read_csv('FimaNfipPolicies.csv', fields=fields):
        df['year_date'] = df['policyEffectiveDate'].str.split('-').str[0]
        df = df[df['year_date'].str.len() == 4]
        df = df[(df['year_date'].astype(int) <= 2100) & (df['year_date'].astype(int) > 1968)]

        df['year_date'] = df['policyTerminationDate'].str.split('-').str[0]
        df = df[df['year_date'].str.len() == 4]
        df = df[(df['year_date'].astype(int) <= 2100) & (df['year_date'].astype(int) > 1968)]

        df['policyEffectiveDate'] = pd.to_datetime(df['policyEffectiveDate'])
        df['policyTerminationDate'] = pd.to_datetime(df['policyTerminationDate'])

        df = df[(df['policyEffectiveDate']  <= doi) & (df['policyTerminationDate'] > doi)]

        pd_dataframes.append(df)

    nfip_data = pd.concat(pd_dataframes)

    # filter outliers
    first_q = nfip_data['totalInsurancePremiumOfThePolicy'].quantile(0.1)
    nineNine_q = nfip_data['totalInsurancePremiumOfThePolicy'].quantile(0.99)
    nfip_data = nfip_data[nfip_data['totalInsurancePremiumOfThePolicy'] > first_q] # greater than 1st percentile
    nfip_data = nfip_data[nfip_data['totalInsurancePremiumOfThePolicy'] < nineNine_q] # less than 99th percentile

    # list of CONUS states
    states = ['AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']

    # extract policies for only CONUS states
    nfip_data = nfip_data[nfip_data['propertyState'].isin(states)]

    # only use policies where the cost exists, field represents only premium (no fees)
    nfip_data = nfip_data[nfip_data['totalInsurancePremiumOfThePolicy'].notna()]
    nfip_data = nfip_data[nfip_data['totalInsurancePremiumOfThePolicy'] > 0]

    # dictionary of CRS class to discount amount
    crs_dict = {1: 0.45, 2: 0.4, 3: 0.35, 4: 0.3, 5: 0.25, 6: 0.2, 7: 0.15, 8: 0.1, 9: 0.05, 10: 0}
    nfip_data.fillna({'crsClassCode':10}, inplace=True)
    # map dictionary of CRS class to discount amount
    nfip_data['CRS_discount'] = nfip_data['crsClassCode'].map(crs_dict)
    # Group policies don't get a CRS discount
    nfip_data.loc[nfip_data['rateMethod'] == 'G', 'CRS_discount'] = 0
    # Provisional policies don't get a CRS discount
    nfip_data.loc[nfip_data['rateMethod'] == '6', 'CRS_discount'] = 0
    ## Communities on probabtion don't get the CRS discount
    nfip_data.loc[nfip_data['communityProbationSurcharge'].isna(), 'communityProbationSurcharge'] = 0
    nfip_data.loc[nfip_data['communityProbationSurcharge'] != 0, 'CRS_discount'] = 0

    nfip_data.to_csv(f"NFIP_crs_{doi}.csv",index=None)

    return


## weighted average function
def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]

    return (d * w).sum() / w.sum()


## calculate the average CRS discount for each state. This would be the average discount percent using the premium as the weight. We can use the historical premium or the full risk premium.
def crs_cross_sub(doi):
    year = doi[:4]

    nfip_data = pd.read_csv(f'NFIP_crs_{doi}.csv', engine="c")

    policies_state_total = nfip_data.groupby('propertyState')['rateMethod'].count().reset_index()
    policies_state_total.rename(columns={'rateMethod':'policiesState'}, inplace=True)

    policies_county = nfip_data.groupby(['countyCode','propertyState'])['rateMethod'].count().reset_index()
    policies_county.rename(columns={'rateMethod':'policiesCounty'}, inplace=True)

    policies_county = policies_county.merge(policies_state_total,on='propertyState')
    policies_county['policy_fraction'] = policies_county['policiesCounty'] / policies_county['policiesState']

    # calculate full premium price based on CRS discount
    nfip_data['premium_noCRS'] = nfip_data['totalInsurancePremiumOfThePolicy'] / (1 - nfip_data['CRS_discount'])

    # use the premium as a weight to calculate the average CRS discount in each county
    crs_weighted = nfip_data.groupby('propertyState').apply(w_avg, 'CRS_discount', 'premium_noCRS', include_groups=False)

    crs_weighted = crs_weighted.to_frame(name='avg_w_crs_disc')
    crs_weighted_dict = crs_weighted.to_dict()['avg_w_crs_disc']

    # Calculate which communities are being subsidized, so which communities have a higher CRS discount than the state average
    nfip_gpd = gpd.read_file('./NFIP_communities/NFIP_communities_CRS.shp')
    states = ['AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
    nfip_gpd = nfip_gpd[nfip_gpd['state'].isin(states)]
    crs_dict = {1: 0.45, 2: 0.4, 3: 0.35, 4: 0.3, 5: 0.25, 6: 0.2, 7: 0.15, 8: 0.1, 9: 0.05, 10: 0, 0: 0}
    # map dictionary of CRS class to discount amount
    nfip_gpd['CRS_discount'] = nfip_gpd['crs_class_'].map(crs_dict)

    # map dictionary of state average CRS discounts
    nfip_gpd['state_crs_avg_discount'] = nfip_gpd['state'].map(crs_weighted_dict)

    # nfip_gpd_subsidizing = nfip_gpd[nfip_gpd['state_crs_avg_discount'] > nfip_gpd['CRS_discount']]
    # nfip_gpd_subsidizing.to_file('./NFIP_communities/NFIP_communities_CRS_subsidizing.shp')

    # Calculate how much more each community and each policyholder in the communities are paying to subsidize the CRS discounted communities
    nfip_data['state_crs_avg_discount'] = nfip_data['propertyState'].map(crs_weighted_dict)

    nfip_data['state_crs_load'] = nfip_data['premium_noCRS'] * nfip_data['state_crs_avg_discount']

    nfip_data['crs_discount_amount'] = nfip_data['premium_noCRS'] - nfip_data['totalInsurancePremiumOfThePolicy']

    nfip_data['net_crs_load'] = nfip_data['state_crs_load'] - nfip_data['crs_discount_amount'] # positive net load means the community is paying extra to subsidize other communities
    if year == 2010:
        nfip_data['net_crs_load']  = nfip_data['net_crs_load'] * 1.4 ## can only convert to 2023 prices, 2024 data is not complete

    print(f"90% range of net CRS load: {nfip_data['net_crs_load'].quantile([0.05,0.95])}")

    print(f"Percent of policies with positive net CRS loads: {nfip_data[nfip_data['net_crs_load'] > 0].shape[0]/nfip_data.shape[0] * 100}%")
    print(f"Percent of policies with negative net CRS loads: {nfip_data[nfip_data['net_crs_load'] <= 0].shape[0]/nfip_data.shape[0] * 100}%")

    check = nfip_data.groupby('propertyState')['net_crs_load'].sum()
    print(check)

    nfip_data['crsLoadpcT'] = nfip_data['net_crs_load'] / nfip_data['totalInsurancePremiumOfThePolicy']
    net_crs_load_community = nfip_data.groupby('countyCode')[['net_crs_load','crsLoadpcT']].mean().reset_index()

    # merge to county shapefile
    county_gpd = gpd.read_file('./cb_2018_us_county_20m/cb_2018_us_county_20m.shp')
    county_gpd['countyCode'] = county_gpd['GEOID'].astype(int)

    nfip_gpd_merged = county_gpd.merge(net_crs_load_community, on='countyCode')
    nfip_gpd_merged.to_file(f'./cb_2018_us_county_20m/cb_2018_us_county_20m_netCRSload_{doi}.shp')

    nfip_gpd_policies = nfip_gpd_merged.merge(policies_county, on='countyCode')
    nfip_gpd_policies.to_file(f'./NFIP_communities/NFIP_communities_policies_{doi}.shp')

    return


## calculate correlation between CRS net load and rural capacity index
def crs_load_correlation():
    nfip_gpd = gpd.read_file('./cb_2018_us_county_20m/cb_2018_us_county_20m_netCRSload.shp')

    nfip_gpd_pos = nfip_gpd[nfip_gpd['net_crs_lo'] > 0]
    nfip_gpd_neg = nfip_gpd[nfip_gpd['net_crs_lo'] < 0]
    print(f"Average positive net CRS load change {nfip_gpd_pos['crsLoadpcT'].mean()*100}%. Average negative net CRS load change {nfip_gpd_neg['crsLoadpcT'].mean()*100}%")
    print(f"Average positive net CRS load ${nfip_gpd_pos['net_crs_lo'].mean()}. Average negative net CRS load ${nfip_gpd_neg['net_crs_lo'].mean()}")
    print(f"Number of positive net CRS load counties: {nfip_gpd_pos.shape[0]}. Number of negative net CRS load counties {nfip_gpd_neg.shape[0]}")
    print(f"Percent of positive net CRS loads between $0 and $100: {nfip_gpd[(nfip_gpd['net_crs_lo'] > 0) & (nfip_gpd['net_crs_lo'] < 100)].shape[0]/nfip_gpd_pos.shape[0]}")
    print(f"Percent of positive net CRS loads between $100 and $200: {nfip_gpd[(nfip_gpd['net_crs_lo'] >= 100) & (nfip_gpd['net_crs_lo'] < 200)].shape[0]/nfip_gpd_pos.shape[0]}")
    print(f"Percent of positive net CRS loads between $200 and $300: {nfip_gpd[(nfip_gpd['net_crs_lo'] >= 200) & (nfip_gpd['net_crs_lo'] < 300)].shape[0]/nfip_gpd_pos.shape[0]}")
    print(f"Percent of positive net CRS loads greater than $300: {nfip_gpd[(nfip_gpd['net_crs_lo'] >= 300)].shape[0]/nfip_gpd_pos.shape[0]}")


    plt.hist(nfip_gpd['net_crs_lo'],bins=np.arange(-500,500,50))
    # use Ginto Normal font
    font_path = '/Users/ddusseau/Documents/Fonts/GintoNormal/GintoNormal-Regular.ttf'  # the location of the font file
    my_font = fm.FontProperties(fname=font_path, size=9)  # get the font based on the font_path
    # set DPI parameter
    plt.rcParams['savefig.dpi'] = 300

    plt.yticks(fontproperties=my_font, fontsize=13)
    plt.xticks(fontproperties=my_font, fontsize=13)
    plt.ylabel("Number of counties" ,fontproperties=my_font, fontsize=13)
    plt.xlabel("Net CRS load ($)" ,fontproperties=my_font, fontsize=13)
    plt.xlim(-300,300)
    plt.savefig("./figures/NetCRSLoad_Histogram.png",bbox_inches='tight')
    plt.show()


    he_rural = pd.read_csv('HE_Rural_Capacity_Index_March_2024_Download_Data.csv')

    nfip_gpd_he = nfip_gpd.merge(he_rural, left_on='countyCode',right_on='FIPS')
    nfip_gpd_he_pos = nfip_gpd_he[nfip_gpd_he['net_crs_lo'] > 0]
    nfip_gpd_he_neg = nfip_gpd_he[nfip_gpd_he['net_crs_lo'] < 0]
    sig_test = st.mannwhitneyu(nfip_gpd_he_pos['RURAL CAPACITY INDEX'], nfip_gpd_he_neg['RURAL CAPACITY INDEX'])
    print(f"Mann Whitney U significance test for net CRS load and rural capacity index p-value: {sig_test[1]}")
    print(f"Mean HE Index Positive CRS: {nfip_gpd_he_pos['RURAL CAPACITY INDEX'].mean()}")
    print(f"Mean HE Index Negative CRS: {nfip_gpd_he_neg['RURAL CAPACITY INDEX'].mean()}")

    nfip_gpd_he_pos_high = nfip_gpd_he_pos[nfip_gpd_he_pos['RURAL CAPACITY INDEX'] > 66]
    print(nfip_gpd_he_pos_high.shape[0]/nfip_gpd_he_pos.shape[0])

    nfip_gpd_he_neg_high = nfip_gpd_he_neg[nfip_gpd_he_neg['RURAL CAPACITY INDEX'] > 66]
    print(nfip_gpd_he_neg_high.shape[0]/nfip_gpd_he_neg.shape[0])

    # plt.scatter(nfip_gpd_he['net_crs_lo'], nfip_gpd_he['RURAL CAPACITY INDEX'])
    # plt.show()


    # median_income = pd.read_csv('Median_Household_Income.csv',skiprows=4)
    # median_income = median_income.replace(',','', regex=True)
    # median_income = median_income[median_income['Median_Household_Income_2021'].notna()]
    # median_income['Median_Household_Income_2021'] = median_income['Median_Household_Income_2021'].astype(int)
    # nfip_gpd_income = nfip_gpd.merge(median_income, left_on='countyCode', right_on='FIPS_Code')
    # nfip_gpd_income_pos = nfip_gpd_income[nfip_gpd_income['net_crs_lo'] > 0]
    # nfip_gpd_income_neg = nfip_gpd_income[nfip_gpd_income['net_crs_lo'] < 0]
    # sig_test = st.mannwhitneyu(nfip_gpd_income_pos['Median_Household_Income_2021'], nfip_gpd_income_neg['Median_Household_Income_2021'], alternative='less')
    # print(sig_test)


    return


## calculate net CRS load under Risk Rating 2.0
def rr2_crs():
    rr2 = pd.read_csv('fema_risk-rating-2.0_exhibit_4.csv', skiprows=2, skipfooter=10)
    rr2.loc[~rr2['Asterisk'].isna(),'Policies in Force (PIF)'] = 5 # change all counties with state average to 5
    rr2['Full Risk Premium'] = rr2['Full Risk Premium'].replace(',','', regex=True).astype(float)
    rr2['Policies in Force (PIF)'] = rr2['Policies in Force (PIF)'].replace(',','', regex=True).astype(float)
    rr2['Average Risk-based Cost of Insurance'] = rr2['Average Risk-based Cost of Insurance'].replace(',','', regex=True).astype(float)
    rr2.loc[~rr2['Asterisk'].isna(),'Full Risk Premium'] = rr2.loc[~rr2['Asterisk'].isna(),'Policies in Force (PIF)'] * rr2.loc[~rr2['Asterisk'].isna(),'Average Risk-based Cost of Insurance']
    rr2_raw = rr2[~rr2['County'].isna()]

    fips_codes = pd.read_csv('all-geocodes-v2020.csv',skiprows=4)
    fips_codes['County Code (FIPS)'] = fips_codes['County Code (FIPS)'].astype(str)
    fips_codes['State Code (FIPS)'] = fips_codes['State Code (FIPS)'].astype(str)
    for index, row in fips_codes.iterrows():
        if len(row['County Code (FIPS)']) == 2:
            fips_codes.at[index,'County Code (FIPS)'] = '0'+row['County Code (FIPS)']
        if len(row['County Code (FIPS)']) == 1:
            fips_codes.at[index,'County Code (FIPS)'] = '00'+row['County Code (FIPS)']
    fips_codes['county_fips'] = fips_codes['State Code (FIPS)'] + fips_codes['County Code (FIPS)']
    fips_codes['county_fips'] = fips_codes['county_fips'].astype(int)
    fips_codes['State Code (FIPS)'] = fips_codes['State Code (FIPS)'].astype(int)

    state_fips = pd.read_csv('state_fips_master.csv')
    fips_codes = fips_codes.merge(state_fips,left_on='State Code (FIPS)',right_on='fips')
    fips_codes.rename(columns={'Area Name (including legal/statistical area description)':'County_Name'}, inplace=True)
    fips_codes = fips_codes[['State Code (FIPS)','county_fips','County_Name','state_name','state_abbr']]
    fips_codes['County_Name'] = fips_codes['County_Name'].str.upper()

    rr2 = rr2_raw.merge(fips_codes, left_on=['State','County'], right_on=['state_abbr','County_Name'], how='inner') #how='left', indicator=True

    nfip_data = pd.read_csv('NFIP_crs_2024-01-31.csv', engine="c")
    nfip_data = nfip_data[nfip_data['occupancyType'].isin([1,11,14])] # single family homes

    crs_discount_fees_avg = nfip_data.groupby('countyCode')[['CRS_discount','reserveFundAssessment','federalPolicyFee']].mean() # average CRS discount, reserve fund assessment, and federal policy fee per county

    rr2_discount = rr2.merge(crs_discount_fees_avg,left_on='county_fips',right_on='countyCode', how='inner') #,how='left',indicator=True

    rr2_discount['total_fees'] = rr2_discount['Policies in Force (PIF)'] * (rr2_discount['reserveFundAssessment'] + rr2_discount['federalPolicyFee'] + 25)
    rr2_discount['Full Risk Premium'] = rr2_discount['Full Risk Premium'] - rr2_discount['total_fees']
    rr2_discount['Average Risk-based Cost of Insurance'] = rr2_discount['Full Risk Premium'] / rr2_discount['Policies in Force (PIF)']
    rr2_discount['total_co_premium_noCRS'] = rr2_discount['Full Risk Premium'] / (1 - rr2_discount['CRS_discount'])

    # use the premium as a weight to calculate the average CRS discount in each county
    crs_weighted = rr2_discount.groupby('State').apply(w_avg, 'CRS_discount', 'total_co_premium_noCRS', include_groups=False)

    crs_weighted = crs_weighted.to_frame(name='avg_w_crs_disc')
    crs_weighted_dict = crs_weighted.to_dict()['avg_w_crs_disc']

    # Calculate how much more each community and each policyholder in the communities are paying to subsidize the CRS discounted communities
    rr2_discount['state_crs_avg_discount'] = rr2_discount['State'].map(crs_weighted_dict)

    rr2_discount['state_crs_load'] = rr2_discount['total_co_premium_noCRS'] * rr2_discount['state_crs_avg_discount']

    rr2_discount['crs_discount_amount'] = rr2_discount['total_co_premium_noCRS'] - rr2_discount['Full Risk Premium']

    rr2_discount['net_crs_load'] = rr2_discount['state_crs_load'] - rr2_discount['crs_discount_amount'] # positive net load means the community is paying extra to subsidize other communities
    rr2_discount['net_crs_load'] = rr2_discount['net_crs_load'] / rr2_discount['Policies in Force (PIF)']

    rr2_discount['crsLoadpcT'] = rr2_discount['net_crs_load'] / rr2_discount['Average Risk-based Cost of Insurance']

    # merge to county shapefile
    county_gpd = gpd.read_file('./cb_2018_us_county_20m/cb_2018_us_county_20m.shp')
    county_gpd['county_fips'] = county_gpd['GEOID'].astype(int)

    nfip_gpd_merged = county_gpd.merge(rr2_discount, on='county_fips')
    nfip_gpd_merged.to_file('./cb_2018_us_county_20m/cb_2018_us_county_20m_netCRSload_RR2.shp')


    return


## calculate correlation between net CRS load and different climate perils in climate vulnerability index
def cvi():
    cvi_data = pd.read_csv('CVI-county-pct-cat-CC-Extreme Events.gis.csv')

    cvi_data[['state', 'county']] = cvi_data['Name'].str.split(', ', expand=True)
    column = 'Droughts!1!0x44aa99ff' #'Wildfires!1!0x117733ff' # #ToxPi Score

    states = cvi_data['state'].unique()
    for s in states:
        state_data = cvi_data[cvi_data['state'] == s]
        state_data['ToxPi Score re norm']=(state_data[column]-state_data[column].min())/(state_data[column].max()-state_data[column].min())
        cvi_data.loc[state_data.index, 'ToxPi Score ReNormalized'] = state_data['ToxPi Score re norm']

    county_gpd = gpd.read_file('./cb_2018_us_county_20m/cb_2018_us_county_20m_netCRSload_RR2.shp')
    county_gpd['FIPS'] = county_gpd['GEOID'].astype(int)

    cvi_gpd_merged = county_gpd.merge(cvi_data, on='FIPS')
    cvi_gpd_merged.to_file('./cb_2018_us_county_20m/cb_2018_us_county_20m_CVI.shp')

    cvi_gpd_merged.dropna(axis=0,subset=['ToxPi Score ReNormalized', 'net_crs_lo'],inplace=True)

    fire_states = ['WA']
    drought_states = ['NC']
    cvi_gpd_merged = cvi_gpd_merged[cvi_gpd_merged['state'].isin(drought_states)]

    plt.scatter(cvi_gpd_merged['net_crs_lo'],cvi_gpd_merged['ToxPi Score ReNormalized'])
    plt.show()

    print(st.pearsonr(cvi_gpd_merged['net_crs_lo'],cvi_gpd_merged['ToxPi Score ReNormalized']))

    return



########################################

doi = '2024-01-31'


preprocess(doi)

crs_cross_sub(doi)

crs_load_correlation()

rr2_crs()

cvi()
