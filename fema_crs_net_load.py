import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob2
import scipy.stats as st

## function to read large CSV file
def read_csv(file_name, fields):
    for chunk in pd.read_csv(file_name, chunksize=1000000, usecols=fields, engine="c"):
        yield chunk

## preprocess NFIP data for date and fields of interest
def preprocess(doi):
    fields = ['propertyState', 'countyCode', 'policyEffectiveDate','crsClassCode','policyTerminationDate', 'occupancyType','nfipRatedCommunityNumber','nfipCommunityNumberCurrent','totalInsurancePremiumOfThePolicy','rateMethod','communityProbationSurcharge','reserveFundAssessment','federalPolicyFee','censusTract']
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

    nfip_data = pd.read_csv(f'NFIP_crs_{doi}.csv', engine="c")

    # calculate full premium price based on CRS discount
    nfip_data['premium_noCRS'] = nfip_data['totalInsurancePremiumOfThePolicy'] / (1 - nfip_data['CRS_discount'])

    # use the premium as a weight to calculate the average CRS discount in each county
    crs_weighted = nfip_data.groupby('propertyState').apply(w_avg, 'CRS_discount', 'premium_noCRS', include_groups=False)

    crs_weighted = crs_weighted.to_frame(name='avg_w_crs_disc')
    crs_weighted_dict = crs_weighted.to_dict()['avg_w_crs_disc']

    # Calculate how much more each community and each policyholder in the communities are paying to subsidize the CRS discounted communities
    nfip_data['state_crs_avg_discount'] = nfip_data['propertyState'].map(crs_weighted_dict)

    nfip_data['state_crs_load'] = nfip_data['premium_noCRS'] * nfip_data['state_crs_avg_discount']

    nfip_data['crs_discount_amount'] = nfip_data['premium_noCRS'] - nfip_data['totalInsurancePremiumOfThePolicy']

    nfip_data['net_crs_load'] = nfip_data['state_crs_load'] - nfip_data['crs_discount_amount'] # positive net load means the community is paying extra to subsidize other communities

    print(f"90% range of net CRS load: {nfip_data['net_crs_load'].quantile([0.05,0.95])}")

    print(f"Percent of policies with positive net CRS loads: {nfip_data[nfip_data['net_crs_load'] > 0].shape[0]/nfip_data.shape[0] * 100}%")
    print(f"Percent of policies with negative net CRS loads: {nfip_data[nfip_data['net_crs_load'] < 0].shape[0]/nfip_data.shape[0] * 100}%")

    check = nfip_data.groupby('propertyState')['net_crs_load'].sum()
    print(check)

    nfip_data['crsLoadpcT'] = nfip_data['net_crs_load'] / nfip_data['totalInsurancePremiumOfThePolicy']

    pos_crs = nfip_data[nfip_data['net_crs_load'] > 0]
    print(f"Average positive net CRS load: {pos_crs['net_crs_load'].mean()}")

    neg_crs = nfip_data[nfip_data['net_crs_load'] < 0]
    print(f"Average negative net CRS load: {neg_crs['net_crs_load'].mean()}")

    nfip_data.to_csv(f'NFIP_crs_{doi}_net_load.csv')

    # merge to county shapefile
    county_gpd = gpd.read_file('./cb_2018_us_county_20m/cb_2018_us_county_20m.shp')
    county_gpd['countyCode'] = county_gpd['GEOID'].astype(int)

    county_net_crs = nfip_data.groupby('countyCode')['net_crs_load'].mean()
    county_net_crs_prct = nfip_data.groupby('countyCode')['crsLoadpcT'].mean()
    nfip_gpd_merged = county_gpd.merge(county_net_crs, on='countyCode')
    nfip_gpd_merged = nfip_gpd_merged.merge(county_net_crs_prct, on='countyCode')
    nfip_gpd_merged.to_file(f'./cb_2018_us_county_20m/cb_2018_us_county_20m_netCRSload_{doi}.shp')

    policies_state_total = nfip_data.groupby('propertyState')['rateMethod'].count().reset_index()
    policies_state_total.rename(columns={'rateMethod':'policiesState'}, inplace=True)

    policies_county = nfip_data.groupby(['countyCode','propertyState'])['rateMethod'].count().reset_index()
    policies_county.rename(columns={'rateMethod':'policiesCounty'}, inplace=True)

    policies_county = policies_county.merge(policies_state_total,on='propertyState')
    policies_county['policy_fraction'] = policies_county['policiesCounty'] / policies_county['policiesState']
    policies_county = county_gpd.merge(policies_county, on='countyCode')
    policies_county.to_file(f'./NFIP_communities/NFIP_communities_policies_{doi}.shp')

    return



## calculate net CRS load under Risk Rating 2.0
def rr2_crs(doi):
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

    nfip_data = pd.read_csv(f'NFIP_crs_{doi}.csv', engine="c")
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


## calculate correlation between CRS net load and rural capacity index
def crs_load_correlation(doi):
    nfip_pd = pd.read_csv(f'NFIP_crs_{doi}_net_load.csv')

    nfip_pd_pos = nfip_pd[nfip_pd['net_crs_load'] > 0]
    nfip_pd_neg = nfip_pd[nfip_pd['net_crs_load'] < 0]
    print(f"Average positive net CRS load change {nfip_pd_pos['crsLoadpcT'].mean()*100}%. Average negative net CRS load change {nfip_pd_neg['crsLoadpcT'].mean()*100}%")
    print(f"Average positive net CRS load ${nfip_pd_pos['net_crs_load'].mean()}. Average negative net CRS load ${nfip_pd_neg['net_crs_load'].mean()}")
    print(f"Number of positive net CRS load policies: {nfip_pd_pos.shape[0]}. Number of negative net CRS load policies {nfip_pd_neg.shape[0]}")
    print(f"Percent of positive net CRS loads between $0 and $100: {nfip_pd[(nfip_pd['net_crs_load'] > 0) & (nfip_pd['net_crs_load'] < 100)].shape[0]/nfip_pd_pos.shape[0]}")
    print(f"Percent of positive net CRS loads between $100 and $200: {nfip_pd[(nfip_pd['net_crs_load'] >= 100) & (nfip_pd['net_crs_load'] < 200)].shape[0]/nfip_pd_pos.shape[0]}")
    print(f"Percent of positive net CRS loads between $200 and $300: {nfip_pd[(nfip_pd['net_crs_load'] >= 200) & (nfip_pd['net_crs_load'] < 300)].shape[0]/nfip_pd_neg.shape[0]}")
    print(f"Percent of positive net CRS loads greater than $300: {nfip_pd[(nfip_pd['net_crs_load'] >= 300)].shape[0]/nfip_pd_pos.shape[0]}")

    print(f"Percent of net CRS loads $100 or greater: {nfip_pd[(nfip_pd['net_crs_load'] >= 100)].shape[0]/nfip_pd.shape[0]}")


    plt.hist(nfip_pd['net_crs_load'],bins=np.arange(-500,500,50))
    # use Ginto Normal font
    font_path = '/Users/ddusseau/Documents/Fonts/GintoNormal/GintoNormal-Regular.ttf'  # the location of the font file
    my_font = fm.FontProperties(fname=font_path, size=9)  # get the font based on the font_path
    # set DPI parameter
    plt.rcParams['savefig.dpi'] = 300

    plt.yticks(fontproperties=my_font, fontsize=13)
    plt.xticks(fontproperties=my_font, fontsize=13)
    plt.ylabel("Number of policies" ,fontproperties=my_font, fontsize=13)
    plt.xlabel("Net CRS load ($)" ,fontproperties=my_font, fontsize=13)
    plt.xlim(-300,300)
    plt.savefig("./figures/NetCRSLoad_Histogram.png",bbox_inches='tight')
    plt.show()


    census_data = pd.read_csv('./DECENNIALDHC2020.P3_2025-01-29T153215/DECENNIALDHC2020.P3-Data.csv',skiprows=1)
    census_data[['geo_prefix', 'countyCode']] = census_data['Geography'].str.split('US',expand=True)
    census_data['countyCode'] = census_data['countyCode'].astype(float)

    var = 'net_crs_load' #'net_crs_load'
    demographic = ' !!Total:!!Black or African American alone'

    nfip_data = nfip_pd.groupby('countyCode')[var].mean()
    nfip_data = pd.DataFrame(nfip_data)

    census_data = census_data.merge(nfip_data,on='countyCode')
    census_data.drop_duplicates(inplace=True)

    census_data.replace('-', '0', inplace=True)
    census_data[demographic] = census_data[demographic].astype(float)
    census_data[demographic] = census_data[demographic] / census_data[' !!Total:']

    # Calculate deciles
    census_data['decile'] = pd.qcut(census_data[demographic], 10, labels=False)

    # Create the figure and axes
    fig, ax = plt.subplots()

    data = []
    for i in census_data['decile'].unique():
        data.append(census_data[census_data['decile'] == i][var])

    ax.boxplot(data)

    correlation_coefficient, p_value = st.pearsonr(census_data[var], census_data[demographic])

    print("Pearson correlation coefficient:", correlation_coefficient)
    print("P-value:", p_value)

    plt.title("Distribution of Net CRS Load for Decile of Percent Black Population")
    plt.ylabel('Net CRS Load ($)')
    plt.xlabel('Decile of Percent Black Population')
    plt.savefig("./figures/NetCRSLoad_BlackPopulation_Boxplot.png",bbox_inches='tight')
    plt.show()


    census_data = pd.read_csv('./DECENNIALDHC2020.P4_2025-01-29T164246/DECENNIALDHC2020.P4-Data.csv',skiprows=1)
    census_data[['geo_prefix', 'countyCode']] = census_data['Geography'].str.split('US',expand=True)
    census_data['countyCode'] = census_data['countyCode'].astype(float)

    var = 'net_crs_load' #'net_crs_load'
    demographic = ' !!Total:!!Hispanic or Latino'

    nfip_data = nfip_pd.groupby('countyCode')[var].mean()
    nfip_data = pd.DataFrame(nfip_data)

    census_data = census_data.merge(nfip_data,on='countyCode')
    census_data.drop_duplicates(inplace=True)

    census_data.replace('-', '0', inplace=True)
    census_data[demographic] = census_data[demographic].astype(float)
    census_data[demographic] = census_data[demographic] / census_data[' !!Total:']

    # Calculate deciles
    census_data['decile'] = pd.qcut(census_data[demographic], 10, labels=False)

    # Create the figure and axes
    fig, ax = plt.subplots()

    data = []
    for i in census_data['decile'].unique():
        data.append(census_data[census_data['decile'] == i][var])

    ax.boxplot(data)

    correlation_coefficient, p_value = st.pearsonr(census_data[var], census_data[demographic])

    print("Pearson correlation coefficient:", correlation_coefficient)
    print("P-value:", p_value)

    plt.title("Distribution of Net CRS Load for Decile of Percent Hispanic Population")
    plt.ylabel('Net CRS Load ($)')
    plt.xlabel('Decile of Percent Hispanic Population')
    plt.savefig("./figures/NetCRSLoad_HispanicPopulation_Boxplot.png",bbox_inches='tight')
    plt.show()


    nfip_data = nfip_pd.groupby('countyCode')[var].mean()
    nfip_data = pd.DataFrame(nfip_data)

    income = pd.read_csv('./ACSST5Y2023.S1901_2025-01-29T151322/ACSST5Y2023.S1901-Data.csv', skiprows=1)
    income[['geo_prefix', 'countyCode']] = income['Geography'].str.split('US',expand=True)
    income['countyCode'] = income['countyCode'].astype(float)

    income = income.merge(nfip_data,on='countyCode')
    var_name = 'Estimate!!Households!!Total'
    income['decile'] = pd.qcut(income[var_name], 10, labels=False)

    # Create the figure and axes
    fig, ax = plt.subplots()

    data = []
    for i in income['decile'].unique():
        data.append(income[income['decile'] == i][var])


    ax.boxplot(data)

    correlation_coefficient, p_value = st.pearsonr(income[var], income[var_name])

    print("Pearson correlation coefficient:", correlation_coefficient)
    print("P-value:", p_value)

    plt.title("Distribution of Net CRS Load for Decile of Median Household Income")
    plt.ylabel('Net CRS Load ($)')
    plt.xlabel('Decile of Median Household Income')
    plt.savefig("./figures/NetCRSLoad_MedianHouseholdIncome_Boxplot.png",bbox_inches='tight')
    plt.show()



    he_rural = pd.read_csv('HE_Rural_Capacity_Index_March_2024_Download_Data.csv')

    nfip_data = nfip_pd.groupby('countyCode')[var].mean()
    nfip_data = pd.DataFrame(nfip_data)

    nfip_gpd_he = nfip_data.merge(he_rural, left_on='countyCode',right_on='FIPS')
    nfip_gpd_he_pos = nfip_gpd_he[nfip_gpd_he['net_crs_load'] > 0]
    nfip_gpd_he_neg = nfip_gpd_he[nfip_gpd_he['net_crs_load'] < 0]
    sig_test = st.mannwhitneyu(nfip_gpd_he_pos['RURAL CAPACITY INDEX'], nfip_gpd_he_neg['RURAL CAPACITY INDEX'])
    print(sig_test)
    print(f"Mann Whitney U significance test for net CRS load and rural capacity index p-value: {sig_test[1]}")
    print(f"Mean HE Index Positive CRS: {nfip_gpd_he_pos['RURAL CAPACITY INDEX'].mean()}")
    print(f"Mean HE Index Negative CRS: {nfip_gpd_he_neg['RURAL CAPACITY INDEX'].mean()}")
    sig_test = st.pearsonr(nfip_gpd_he['RURAL CAPACITY INDEX'], nfip_gpd_he['net_crs_load'])
    print(f"Pearson Correlation value: {sig_test[0]} and p-value:{sig_test[1]}")

    nfip_gpd_he_pos_high = nfip_gpd_he_pos[nfip_gpd_he_pos['RURAL CAPACITY INDEX'] > 66]
    nfip_gpd_he_neg_high = nfip_gpd_he_neg[nfip_gpd_he_neg['RURAL CAPACITY INDEX'] > 66]
    print(f'Counties with negative CRS load are {(nfip_gpd_he_neg_high.shape[0]/nfip_gpd_he_neg.shape[0])/(nfip_gpd_he_pos_high.shape[0]/nfip_gpd_he_pos.shape[0])} more likely to have a high capacity')

    return



########################################

if __name__ == "__main__":

    doi = '2025-01-01'


    # preprocess(doi)

    # crs_cross_sub(doi)

    # rr2_crs(doi)

    crs_load_correlation(doi)
