- This project is a Python program that analyzes Titanic passenger data and predicts survival based on various features. It is designed for learning purposes and data analysis practice.


## Features
- Load and clean Titanic dataset
- Analyze numerical and categorical features
- Visualize data using plots
- Predict survival using simple models





#image-1
* Most passengers didnot survive
* The 3rd class was the most populated followed by 1st and 2nd
* There were significantly more males than females
* The vast majority of passengers embarked from Southampton
* Most passengers travelled alone

#image-2
* The distribution peaks around the 20-30 age range. Remember we filled missing values with the median (28), which contributes to the height of that central bar. 
* The distribution is heavily right-skewed, confirming that most tickets were cheap, with a few very expensive exceptions.

#image-3
* A clear trend emerges: 1st class passengers had a >60% survival rate, while 3rd class passengers had less than 25%.
* This is the strongest predictor. Females had a survival rate of ~75%, while males had a rate below 20%.
* Passengers embarking from Cherbourg ('C') had a higher survival rate than those from the other ports.
* Passengers with a registered cabin number had a much higher survival rate. This is likely correlated with being in 1st class.

#image-4 
* Infants and young children had a higher probability of survival.
* A large portion of non-survivors were young adults (20-40).

#image-5
* The box plot confirms the presence of significant outliers. Most fares are concentrated below \$100, but there are several fares extending far beyond,
  with some even exceeding \$500. These are likely first-class passengers who booked luxurious suites.For some machine learning models,handling these outliers
  (e.g., through log transformation) would be an important step.

#image-6
* Passengers who were alone (`IsAlone=1`) had a lower survival rate (~30%) than those in small families.
* Very large families (5 or more) had a very poor survival rate. This might be because it was harder for large families to stay together and evacuate.

#image-7
* The `Title` feature gives us powerful information. 'Mrs' and 'Miss' (females) had high survival rates. 'Mr' (males) had a very low survival rate. 'Master' (young boys)
  had a significantly higher survival rate than 'Mr', reinforcing the 'children first' idea. The 'Rare' titles, often associated with nobility or status, also had a mixed
  but generally higher survival rate than common men.

