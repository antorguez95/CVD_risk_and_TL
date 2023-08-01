import pandas as pd 
from typing import Tuple, List
import os
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

def cat2num_Inger(data: pd.DataFrame):
    """This function replaces the Inger in the Inger database by the KNN
    Imputer, since this tool works better with 
    numbers. For more information, check 
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html. 
    It returns a DataFrame after this replacement. 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['age_grp_SSB']  = data['age_grp_SSB'].replace(['35-39','55-59','65-69','50-54','60-64', 
                                '40-44','45-49','30-34','20-24','25-29','16-19'],
                                [0,1,2,3,4,5,6,7,8,9,10])
    data['age_grp_xtra']  = data['age_grp_xtra'].replace(['30-39','50-59','60-69','40-49','16-29'],
                                [0,1,2,3,4])
    data['sex']  = data['sex'].replace(['man','woman'],[0,1])
    data['marital_status']  = data['marital_status'].replace(['married/cohabit','single','divorced',
                                'widow/widower'],
                                [0,1,2,3])
    data['county_names']  = data['county_names'].replace(['Agder','Innlandet','Møre og Romsdal','Nordland',
                        'Oslo','Rogaland','Telemark and Vestfold','Troms and Finmark','Trøndelag',
                        'Vestlandet','Viken'],
                        [0,1,2,3,4,5,6,7,8,9,10])
    data['bmi_4grps']  = data['bmi_4grps'].replace(['overweight','healthy weight','obese','underweight'],
                                [0,1,2,3])
    data['education']  = data['education'].replace(['college/university less than 4 years',
                        'high school minimum 3 years','primary school up to 10 years',
                        'college/university 4 or more years'],
                                [0,1,2,3])
    data['smoking_hx']  = data['smoking_hx'].replace(['never','former daily','former occasional',
                                'current daily','current ocasional'],
                                [0,1,2,3,4])
    data['snuff_use_hx']  = data['snuff_use_hx'].replace(['never','current daily','former occasional',
                        'former daily','current ocasional'],
                                [0,1,2,3,4])
    data['e_cigarette_use_hx']  = data['e_cigarette_use_hx'].replace(['never','current daily','former daily',
                                'current ocasional','former occasional'],
                                [0,1,2,3,4])
    data['alcoh_drink_freq']  = data['alcoh_drink_freq'].replace(['≤1 per month','2-3 times per week',
                                '≥4 times per week','2-4 times per month'],
                                [0,1,2,3])
    data['alcoh_units']  = data['alcoh_units'].replace(['1-2 units','5-6 units','7-9 units','3-4 units','10+ units'],
                                [0,1,2,3,4])
    data['six_units_or_more_frequ']  = data['six_units_or_more_frequ'].replace(['never','monthly','less than monthly',
                                'weekly','dly or almost dly'],
                                [0,1,2,3,4])
    data['strenous_phy_activ2']  = data['strenous_phy_activ2'].replace(['0','3-4','5-6','1-2','7'],
                                [0,1,2,3,4])
    data['moder_phys_activ2']  = data['moder_phys_activ2'].replace(['1-2','3-4','5-6','7','0'],
                                [0,1,2,3,4])
    data['walking_2']  = data['walking_2'].replace(['1-2','7','3-4','5-6','0'],
                                [0,1,2,3,4])
    data['dly_sitting_hrs2']  = data['dly_sitting_hrs2'].replace(['6-8','0-2','3-5','9-11','12-14','≥15'],
                                [0,1,2,3,4,5]) 
    data['xtra_salt']  = data['xtra_salt'].replace(['occasionally','never','often','always'],
                                [0,1,2,3])
    data['house_income1']  = data['house_income1'].replace(['kr751,000-1,000,000','kr351,000-450,000',
                        'Over kr1,000,000','kr451,000-550,000','kr551,000-750,000','kr251,000-350,000',
                        'kr150,000-250,000','under kr150,000'],
                        [0,1,2,3,4,5,6,7])
    data['house_income2']  = data['house_income2'].replace(['kr751-1,000,000','kr351-550,000',
                        'over kr1,000,000','kr551-750,000','kr150-350,000','under 150'],
                        [0,1,2,3,4,5])
    data['work_life_status_N']  = data['house_income2'].replace(['full-time job','unemployed','age-retired pensioner',
                                'sick leave','part-time job','student','social security benefit','disability benefit',
                                'unemployment benefit','house-/home-keeping','military service'],
                                [0,1,2,3,4,5,6,7,8,9,10])

    return data

def prepare_Inger(dataset_path : str = "", filename : str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Read the Inger dataset from a .csv file and suit it to be processed 
    as a pd.DataFrame. This converted DataFrame is returned. 

    Args:
    -----
            dataset_path: path where dataset is stored. Set by default.
            filename : file name of the .csv containing the dataset. Set by default.

    Returns:
    --------
            data: dataframe containing the whole dataset
            X : dataframe containing the dataset features
            Y : dataframe containing only the target variable
            cols_names: list of strings containing feature names. 
            y_tag: string containing target variable name.
    """

    # Go to dataset path
    os.chdir(dataset_path)

    # Open the .csv file and convert it into DataFrame
    data = pd.read_csv(filename)

    # Store column names 
    cols_names = data.columns

    # Convert "Yes" and "No" values into "1" and "0"
    data.replace(('yes', 'no'), (1, 0), inplace=True)
    
    # Replace nan values by np.nan and 
    data.replace(('nan'), (np.nan), inplace=True)
    data.replace(('NaN'), (np.nan), inplace=True)
    
    # Drop useless rows and columns 
    data = data.drop(['id'], axis=1) 

    # Replace expressions by their equivalent numerical values
    data['strenous_phy_activ1']  = data['strenous_phy_activ1'].replace(['0 days','4 days','5 days','6 days','3 days',
                                '1 day','2 days','7 days'],
                                [0,4,5,6,3,1,2,7])  
    data['moder_phys_activ1']  = data['moder_phys_activ1'].replace(['1 day','4 days','2 days','3 days','6 days',
                                '5 days','7 days','0 days'],
                                [1,4,2,3,6,5,7,0]) 
    data['walking_1']  = data['walking_1'].replace(['1 day','7 days','3 days','6 days','0 days','2 days','4 days', '5 days'],
                                [1,7,3,6,0,2,4,5]) 
    data['dly_sitting_hrs1']  = data['dly_sitting_hrs1'].replace(['8','2','4','10 siiting hours','3','5','12','7','18','9','13',
                                '11','6','1','14','15','16','20 sitting hours','zero sitting hours','17'],
                                [8,2,4,10,3,5,12,7,18,9,13,11,6,1,14,15,16,20,0,17]) 
    data['fish_fishmeal1']  = data['fish_fishmeal1'].replace(['1 time','3 times','2 times','0 times','4 times','5 times','7 times',
                                '6 times'],
                                [1,3,2,0,4,5,7,6])   
    ###### Could be controversial. Check with physicians
    data['sugary_drinks']  = data['sugary_drinks'].replace(['1 glass','2 glasses','0 intake','5 glasses','3 glasses','4 glasses',
                                '6 glasses','≥7 glasses'],
                                [1,2,0,5,3,4,6,7])
    data['fruits_berries1']  = data['fruits_berries1'].replace(['0 intake','2','1','3','4','5','6','7','8','9','10+ intake'],
                                [0,2,1,3,4,5,6,7,8,9,10])
    data['fruits_berries2']  = data['fruits_berries2'].replace(['0','2','1','3','4','≥5'],
                                [0,2,1,3,4,5])
    data['lettuce_veg1']  = data['lettuce_veg1'].replace(['0 intake','1','2','3','4','5','10+ intake','7','6','8'],
                                [0,1,2,3,4,5,10,7,6,8])
    data['lettuce_veg2']  = data['lettuce_veg2'].replace(['0','1','2','3','4','≥5'],
                                [0,1,2,3,4,5])
    data['red_meat1']  = data['red_meat1'].replace(['3','2','4','1','5','6','7 intake','0 intake'],
                                [3,2,4,1,5,6,7,0])
    data['red_meat2']  = data['red_meat2'].replace(['3','2','4','1','≥5','0'],
                                [3,2,4,1,5,0])
    data['pro_meat2']  = data['pro_meat2'].replace(['3','1','0','2','4','≥5'],
                                [3,1,0,2,4,5])
    data['household_adults1']  = data['household_adults1'].replace(['2','1','0','3','4','7','5','6','10+'],
                                [2,1,0,3,4,7,5,6,10])
    data['household_adults2']  = data['household_adults2'].replace(['2','1','0','3','≥4'],
                                [2,1,0,3,4])
    data['household_youngs1']  = data['household_youngs1'].replace(['2','0','3','1','4','5','10+','6'],
                                [2,0,3,1,4,5,10,6])
    data['household_youngs2']  = data['household_youngs2'].replace(['2','0','3','1','≥4'],
                                [2,0,3,1,4])
    ##### End of "controversy"
    data['pro_meat1']  = data['pro_meat1'].replace(['3 times','1 time','0 times','2 times','4 times','6 times','7 times',
                        '5 times'],
                                [3,1,0,2,4,6,7,5])
    data['fish_fishmeal2']  = data['fish_fishmeal2'].replace(['1','3','2','0','4','≥5'],
                                [1,3,2,0,4,5])


    # Converts from category to numerical values 
    data = cat2num_Inger(data)

    # Drop columns not useless, but that might be redundant (this might change with iterations)
    # Columns represented numerical variables but with less values are removed 
    data = data.drop(['fruits_berries2','lettuce_veg2', 'red_meat2','household_adults2',
                'household_youngs2','fish_fishmeal2','height_in_m', 'dly_sitting_hrs2'], axis=1)

    # Alcohol binary variable could be enough 
    data = data.drop(['alcoh_units','six_units_or_more_frequ'], axis=1)

    # Columns represented categorical variables but with more categories are removed
    data = data.drop(['age_grp_SSB', 'bmi_4grps','bmi_6grps', 'alcoh_drink_freq', 'strenous_phy_activ2',
    'moder_phys_activ2','walking_2', 'house_income1','counties','pro_meat2'], axis=1) # Check bmi4grps/6, alcoh_drink_freq with physicians
     
    # This code is designed to classify CVD existance or not, the rest of the studied diseases are drop
    data = data.drop(['high_BP_yn','high_chol_yn','arterial_fib_yn','myocardiac_infarc_yn',
                        'heart_failure_yn','stroke_yn','COPD_yn','DM_yn','cancer_yn'], axis=1)
    
    # Store features' and target variable's names 
    cols_names_prev = data.columns
    y_tag = cols_names_prev[len(cols_names_prev)-1]
    cols_names = cols_names_prev[0:cols_names_prev.size]

    # Save X, Y, feature names and Y name 
    y_tag = cols_names[len(cols_names)-1]
    cols_names = cols_names[0:len(cols_names)-1]
    X = data[cols_names]
    Y = data[y_tag]
    
    return data, X, Y, cols_names, y_tag

def numerical_conversion_Inger(data : np.array, features : str, y_col : str):
    """Fix all Inger database features data types to its original type after KNNImputer is used,
    since this functions returns only a floating points ndarray. For more, check sklearn 
    documentation of this function at
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html. After 
    fixing datatypes, an ndarray to pd.DataFrame conversion is performed. Notice that this
    operation is only done in the fields that were not originally floats.

    Args:
    -----
            data: data returned by KNN Imputer (float data type).
            features: list of strings containing the feature names of the dataset. 
            y_col: target variable (i.e., Y) name 

    Returns:
    --------
            data: dataframe containing the whole dataset after imputation
            X : dataframe containing the dataset features after imputation
            y : dataframe containing only the target variable after imputation
    """
    # From ndarray to pd.DataFrame
    names = features.insert(len(features), y_col)
    data = pd.DataFrame(data, columns = names)
    
    # Fixing necessary datatypes to int (including categorical variables)
    data['age_grp_xtra'] = data['age_grp_xtra'].astype(int)
    data['sex'] = data['sex'].astype(int)
    data['marital_status'] = data['marital_status'].astype(int)
    data['county_names'] = data['county_names'].astype(int)
    data['height'] = data['height'].astype(int)
    data['weight'] = data['weight'].astype(int)
    data['bmi'] = data['bmi'].astype(float)
    data['education'] = data['education'].astype(int)
    data['smoking_hx'] = data['smoking_hx'].astype(int)
    data['snuff_use_hx'] = data['snuff_use_hx'].astype(int)
    data['e_cigarette_use_hx'] = data['e_cigarette_use_hx'].astype(int)
    data['alcohol'] = data['alcohol'].astype(int)
    data['strenous_phy_activ1'] = data['strenous_phy_activ1'].astype(int)
    data['moder_phys_activ1'] = data['moder_phys_activ1'].astype(int)
    data['walking_1'] = data['walking_1'].astype(int)
    data['dly_sitting_hrs1'] = data['dly_sitting_hrs1'].astype(int)
    data['xtra_salt'] = data['xtra_salt'].astype(int)
    data['sugary_drinks'] = data['sugary_drinks'].astype(int)
    data['fruits_berries1'] = data['fruits_berries1'].astype(int)
    data['lettuce_veg1'] = data['lettuce_veg1'].astype(int)
    data['red_meat1'] = data['red_meat1'].astype(int)
    data['pro_meat1'] = data['pro_meat1'].astype(int)
    data['fish_fishmeal1'] = data['fish_fishmeal1'].astype(int)
    data['house_income2'] = data['house_income2'].astype(int)
    data['household_adults1'] = data['household_adults1'].astype(int)
    data['work_life_status_N'] = data['work_life_status_N'].astype(int)
    data['CVD_yn'] = data['CVD_yn'].astype(int)
    
    # Separate X and Y 
    X = data[features]
    y = data[[y_col]]    
     
    return data, X, y

def num2cat_Inger(data : pd.DataFrame):
    """This function replaces the numerical values corresponding to categories in 
    the Inger database by its correspondant category. It returns a DataFrame
    after this replacement. Notice that there are parts of the code that are 
    commented, since through iterations in the pre-processing of this databases,
    variables used might change. 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
#     data['age_grp_SSB']  = data['age_grp_SSB'].replace([0,1,2,3,4,5,6,7,8,9,10],
#                                 ['35-39','55-59','65-69','50-54','60-64', 
#                                 '40-44','45-49','30-34','20-24','25-29','16-19']
#                                 )
    data['age_grp_xtra']  = data['age_grp_xtra'].replace([0,1,2,3,4],
                                ['30-39','50-59','60-69','40-49','16-29']
                                )
    data['sex']  = data['sex'].replace([0,1],['man','woman'])
    data['marital_status']  = data['marital_status'].replace([0,1,2,3],
                                ['married/cohabit','single','divorced',
                                'widow/widower']
                                )
    data['county_names']  = data['county_names'].replace([0,1,2,3,4,5,6,7,8,9,10],
                        ['Agder','Innlandet','Møre og Romsdal','Nordland',
                        'Oslo','Rogaland','Telemark and Vestfold','Troms and Finmark','Trøndelag',
                        'Vestlandet','Viken']
                        )
#     data['bmi_4grps']  = data['bmi_4grps'].replace([0,1,2,3],
#                         ['overweight','healthy weight','obese','underweight']
#                                 )
    data['education']  = data['education'].replace([0,1,2,3], 
                        ['college/university less than 4 years',
                        'high school minimum 3 years','primary school up to 10 years',
                        'college/university 4 or more years']
                                )
    data['smoking_hx']  = data['smoking_hx'].replace([0,1,2,3,4],
                                ['smoke_never','smoke_former_daily','smoke_former_occasional',
                                'smoke_current_daily','smoke_current_ocasional']
                                )
    data['snuff_use_hx']  = data['snuff_use_hx'].replace([0,1,2,3,4],
                                ['snuff_never','snuff_current_daily','snuff_former_occasional',
                        'snuff_former_daily','snuff_current_ocasional']
                                )
    data['e_cigarette_use_hx']  = data['e_cigarette_use_hx'].replace([0,1,2,3,4],
                                ['e_cigar_never','e_cigar_current daily','e_cigar_former daily',
                                'e_cigar_current_ocasional','e_cigar_former_occasional']
                                )
#     data['alcoh_drink_freq']  = data['alcoh_drink_freq'].replace([0,1,2,3],
#                                 ['≤1 per month','2-3 times per week',
#                                 '≥4 times per week','2-4 times per month']
#                                 )
#     data['alcoh_units']  = data['alcoh_units'].replace([0,1,2,3,4],
#                                 ['1-2 units','5-6 units','7-9 units','3-4 units','10+ units'],
#                                 )
#     data['six_units_or_more_frequ']  = data['six_units_or_more_frequ'].replace([0,1,2,3,4],
#                                 ['never','monthly','less than monthly',
#                                 'weekly','dly or almost dly']
#                                 )
#     data['strenous_phy_activ2']  = data['strenous_phy_activ2'].replace([0,1,2,3,4],
#                                 ['0','3-4','5-6','1-2','7'],
#                                 )
#     data['moder_phys_activ2']  = data['moder_phys_activ2'].replace([0,1,2,3,4],
#                                 ['1-2','3-4','5-6','7','0'],
#                                 )
#     data['walking_2']  = data['walking_2'].replace([0,1,2,3,4],
#                                 ['1-2','7','3-4','5-6','0']
#                                 )
#     data['dly_sitting_hrs2']  = data['dly_sitting_hrs2'].replace([0,1,2,3,4,5],
#                                 ['6-8','0-2','3-5','9-11','12-14','≥15']
#                                 ) 
    data['xtra_salt']  = data['xtra_salt'].replace([0,1,2,3],
                        ['salt_occasionally','salt_never','salt_often','salt_always']
                                )
#     data['house_income1']  = data['house_income1'].replace([0,1,2,3,4,5,6,7],
#                                 ['kr751,000-1,000,000','kr351,000-450,000',
#                         'Over kr1,000,000','kr451,000-550,000','kr551,000-750,000','kr251,000-350,000',
#                         'kr150,000-250,000','under kr150,000']
#                         )
    data['house_income2']  = data['house_income2'].replace([0,1,2,3,4,5],
                        ['kr751-1,000,000','kr351-550,000',
                        'over kr1,000,000','kr551-750,000','kr150-350,000','under 150']
                        )
    data['work_life_status_N']  = data['work_life_status_N'].replace([0,1,2,3,4,5,6,7,8,9,10],
                                ['full-time job','unemployed','age-retired pensioner',
                                'sick leave','part-time job','student','social security benefit','disability benefit',
                                'unemployment benefit','house-/home-keeping','military service']
                                )
 
    return data 
