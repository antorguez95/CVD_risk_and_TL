# Copyright (C) 2023 Antonio Rodriguez
# 
# This file is part of Transfer_Learning_for_CVD_risk_calculators.
# 
# Transfer_Learning_for_CVD_risk_calculators is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Transfer_Learning_for_CVD_risk_calculators is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Transfer_Learning_for_CVD_risk_calculators.
# If not, see <http://www.gnu.org/licenses/>.

# Dataset paths
FRAM_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\Bases de datos\Diabetes\Framingham"
STENO_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\Bases de datos\Diabetes\STENO DMT1"
INGER_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\Bases de datos\Diabetes\Inger"

# File names 
fram_filename = "framingham_data.csv"
steno_filename1 = "sampleData.csv"
steno_filename2 = "stenoRiskReport.csv"
inger_filename = "WARIFAdataset_v2.csv"

# Dataset names
fram_name = 'FRAMINGHAM'
steno_name = 'STENO'
inger_name = 'INGER'

# Declaration of numerical and categorical features  
fram_numerical_features = ['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
steno_numerical_features = ['age','dm','sbp','ldl','hba1c','egfr']
inger_numerical_features = ['height','weight','bmi','strenous_phy_activ1','moder_phys_activ1','walking_1',
                            'dly_sitting_hrs1','sugary_drinks','fruits_berries1','lettuce_veg1','red_meat1',
                            'pro_meat1','fish_fishmeal1','household_adults1','household_youngs1']

fram_categorical_features = ['education']
steno_categorical_features = ['albuminuria']
inger_categorical_features = ['age_grp_xtra', 'sex', 'marital_status', 'county_names', 'education', 'smoking_hx', 
                        'snuff_use_hx', 'e_cigarette_use_hx', 'xtra_salt', 'house_income2','work_life_status_N']