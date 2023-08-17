# Copyright (C) 2023 Antonio Rodriguez
# 
# This file is part of CVD_risk_and_TL.
# 
# CVD_risk_and_TL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# CVD_risk_and_TL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CVD_risk_and_TL.
# If not, see <http://www.gnu.org/licenses/>.

# Dataset paths
FRAM_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\Bases de datos\Diabetes\Framingham"
STENO_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\Bases de datos\Diabetes\STENO DMT1"

# File names 
fram_filename = "framingham_data.csv"
steno_filename1 = "sampleData.csv"
steno_filename2 = "stenoRiskReport.csv"

# Dataset names
fram_name = 'FRAMINGHAM'
steno_name = 'STENO'

# Declaration of numerical and categorical features  
fram_numerical_features = ['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
steno_numerical_features = ['age','dm','sbp','ldl','hba1c','egfr']

fram_categorical_features = ['education']
steno_categorical_features = ['albuminuria']
