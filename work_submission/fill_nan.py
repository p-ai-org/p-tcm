# Eric

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

ii_imp = IterativeImputer(
    estimator=ExtraTreesRegressor(), max_iter=10, random_state=1121218
)

inputs = df.drop("SVM Training", axis = 1)
# drop whatever output columns you have, we only want to fit our data onto inputs

inputs = ii_imp.fit_transform(inputs)

new_data = pd.DataFrame(inputs, columns = ['Water (g)',
 'Energy (kal)',
 'Protein (g)',
 'lipid (g)',
 'Carbohydrate (g)',
 'Fiber (g)',
 'Sugars (g)',
 'Ash (g)',
 'Ca (mg)',
 'Fe (mg)',
 'Mg (mg)',
 'P (mg)',
 'K (mg)',
 'Na (mg)',
 'Zn (mg)',
 'Se (µg)',
 'Cu (mg)',
 'Mn (mg)',
 'I (µg)',
 'Vc (mg)',
 'Thiamin (mg)',
 'Riboflavin (mg)',
 'Niacin (mg)',
 'B6 (mg)',
 'Folate,DFE (µg)',
 'B12 (µg)',
 'Va,RAE (µg)',
 'Ve (mg)',
 'Vd (IU)',
 'Vk (µg)',
 'saturated (g)',
 'monounsaturated (g)',
 'polyunsaturated (g)',
 'trans (g)',
 'Cholesterol (mg)',
 'Caffeine (mg)',
 'phenolics (mg)',
 'pH'])

# return inputs, which is now an array of values, back into a csv with column names
