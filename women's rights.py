#!/usr/bin/env python
# coding: utf-8

# DATASET USED: World Bank Gender Statistics (2025) – processed by Our World in Data.
# “Employment discrimination based on gender prohibited - Yes (Count)” [dataset].
# World Bank Gender Statistics, “World Bank Gender Statistics” [original data].  Please find this here:  https://raw.githubusercontent.com/Helencrowtherlib/Everything_counts_asst2_HC/refs/heads/main/number-of-countries-with-key-economic-and-social-rights-for-women.csv
# 
# AI STATEMENT: Generative AI has been used to generate code for data cleaning, plotting, and statistical tests.
# 
# This dataset contains the number of countries that have laws in place to ensure women's rights with resepect to:employment discrimination, equal pay, paid msternity leave, domestic violence, and property rights.  It tracks the addition of legislation year by year, as countries adopted this legislation, beginning in 1970. The IMF imposes conditionalities on countries in respect of improving gender equailty, such that it estimates about 280 pieces of legislation were passed between 1960 and 2010, following the UN Charter and Conventions on women's rights and those of children(Christopherson Puh et al., 2022) This coincides with second-wave feminist movement, and the development of global governance in the shape of the UN and IMF post-war.
# 
# These rights have become normative factors and thus act to diffuse women's rights and legitimise gender equality.  
# This data tracks the rate of adoption of these various rights, and it therefore becomes useful to identify the timescales of the adoption of each broad category of legislation, in particular if they show the same kind of rates of adoption, and the relations (if any) between them. 
# 
# 
# 

# In[19]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

df_url = 'https://raw.githubusercontent.com/Helencrowtherlib/Everything_counts_asst2_HC/refs/heads/main/number-of-countries-with-key-economic-and-social-rights-for-women.csv'
url_content = requests.get(df_url, verify = False).content
women = pd.read_csv(io.StringIO(url_content.decode('utf-8')))
women


# DATA CLEANING

# In[27]:


#Reset the index to numbers - there ought to be a range index because that's what pandas uses
women = women.reset_index(drop=True)


# In[28]:


#Rename columns for clarity
women = women.rename(columns={
    "level_0": "entity",
    "level_1": "code",
    "level_2": "year",
    "level_3": "employment_discrimination_prohibited",
    "level_4": "equal_pay_law",
    "level_5": "paid_maternity_leave_14w",
    "level_6": "domestic_violence_legislation",
    "level_7": "equal_property_rights",
    "number-of-countries-with-key-economic-and-social-rights-for-women": "country_count"
})


# In[30]:


women = women[women["entity"] != "Entity"].copy()

women["year"] = pd.to_numeric(women["year"], errors="coerce")
women["country_count"] = pd.to_numeric(women["country_count"], errors="coerce")
women = women.dropna(subset=["year", "country_count"]).copy()



# In[31]:


women.columns


# In[12]:


#Check how the table looks now
women.info()
women.head()


# In[66]:


#Identify the "rights" columns 
rights_cols = [
    "employment_discrimination_prohibited",
    "equal_pay_law",
    "paid_maternity_leave_14w",
    "domestic_violence_legislation",
    "equal_property_rights"
]
id_vars = ["entity", "code", "year", "country_count"]

rights_cols = [c for c in women.columns if c not in id_vars]

#Melt (this turns the wide format to a long one)
women_long = women.melt(
    id_vars=["entity", "code", "year", "country_count"],
    value_vars=rights_cols,
    var_name="right",
    value_name="count_with_right")

women_long["count_with_right"] = pd.to_numeric(
    women_long["count_with_right"], errors="coerce")


#Sanity check
women_long.head(10)


# In[68]:


#Work out percentages of countries adopting rights
women_long["pct_with_right"] = (women_long["count_with_right"] / women_long["country_count"]) * 100

#Bin the 'world' aggregate because otherwise it will be treated as an entity to be calculated and skew the data
women_long = women_long[women_long["entity"] != "World"]


# In[69]:


#Check the datatypes are correct
women_long.head(3)
women_long.dtypes


# In[72]:


#Make a line plot so we can see what rates of adoption look like compared with each other

plt.figure(figsize=(10, 6))
plt.ylim(0, 100)

sns.lineplot(
    data=women_long,
    x="year",
    y="pct_with_right",
    hue="right"
)

plt.xlabel("Year")
plt.ylabel("Percentage of countries with right")
plt.title("Global Adoption of Women's Rights Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

Fig 1 (above)
# In[73]:


#Pretty line plot, but do facets for clarity
g = sns.FacetGrid(women_long, col="right", col_wrap=3, height=3)
g.map_dataframe(sns.lineplot, x="year", y="pct_with_right")
g.set_axis_labels("Year", "% of countries")
g.tight_layout()

Fig 2Figs 1 and 2 (above)
These descriptive trends vary substantially across rights. Domestic violence legislation shows the fastest rate of adoption, followed by employment discrimination prohibitions. Equal pay laws show a moderate rate of diffusion, while paid maternity leave extends more slowly. Equal property rights display the slowest rate of adoption over time. The 'equal property rights' shows higher adoption rates at 1970 than all of the other rights, suggesting that equal property rights had high baseline adoption, so presumably subsequent growth was slower (because there were already many countries with rights of this sort).  In this respect, the data indicates the historical adoption of categories of rights. Tertilt, et al (2022, p 3) report that the progress of women's rights begins with economic rights (property in this dataset), followed by equlaity in the labour market (equal pay law, employment discrimination, maternity leave), followed last of all by rights over one's own body (domestic violence laws).

Formal hypothesis testing is required to assess whether these differences *exceed* what would be expected under shared temporal diffusion given that many countries have adopted legal protections,  encouraged / enforced by glabal governance. 
Also, there is a long time span in the data which permits more data points and thus we can visualise coherent trajectories; and all the rights show an upward trajectory, which is intriguing enough to invite formal testing.  
# In[74]:


#Make a matrix so we can see how closely correlated the different rights are
women_pivot = women_long.pivot_table(
    index="year",
    columns="right",
    values="pct_with_right"
)
women_pivot.corr()

Fig 3 (above)  It can be seen that there is high correlation between the different rights trajectories.  We could expect to see this multicollinearity in an environment where women's rights are related and constrained by women's campaigns and global governance.  There is least correlation between domestic violence legislation and property rights, but this would reflect a difference in civil/criminal law, family/personal rights and property ownership.  At all events, the overall high correlation suggests rights diffusion at similar rates, and therefore worthy of investigation.HYPOTHESIS:
H0: women's rights diffuse at the same rate over time, even if they start from different baselines (null hypothesis)
HA: different women's rights diffuse at different rates over time (alternative hypothesis)


# In[48]:


#REGRESSION

# H0 model: same slope over time for all rights, different intercepts allowed
m0 = smf.ols("pct_with_right ~ year + C(right)", data=women_long).fit()

# HA model: different slopes over time for each right (interaction)
m1 = smf.ols("pct_with_right ~ year * C(right)", data=women_long).fit()

print("MODEL 0\n")
print(m0.summary())
print("\n" + "-"*80 + "\n")
print("MODEL 1\n")
print(m1.summary())

Figs 3 and 4 (above)
The coefficients associated with each right differ in size and interaction with time, showing differing diffusion rates across different global regions. Under H0, which allows for different baseline adoption levels but constrains all rights to a common diffusion rate, about 59% of the variation in adoption percentages is explained by time and right-specific intercepts. Under HA, which allows diffusion rates to vary across rights, this increases to 67% of the variation explained, showing an improvement in model fit.
# In[75]:


F_stat, p_val, df_diff = m1.compare_f_test(m0)

print(f"F-test for equal slopes:")
print(f"  F({int(df_diff)}, df_resid) = {F_stat:.2f}")
print(f"  p-value = {p_val:.2e}")

F-test is the formal hypothesis test.  To test the null hypothesis that women’s rights diffuse at a common rate, a model with a shared time trend is compared to a model allowing right-specific slopes. An F-test rejects the null hypothesis of equal diffusion rates (F = 103.85, p < 0.001). The large F-statistic reflects a substantial reduction in unexplained variation when allowing right-specific diffusion rates, relative to the residual variability in the data.
# In[51]:


m1 = smf.ols(
    "pct_with_right ~ year * C(right, Treatment(reference='employment_discrimination_prohibited'))",
    data=women_long
).fit()


# In[76]:


#Regression plot
#Counterfactual is the H0  - how adoption rates would look if all countries adopted rights at the same rate

plt.figure(figsize=(10, 6))

years = np.linspace(women_long["year"].min(), women_long["year"].max(), 100)

# Full model (H1)
for r in women_long["right"].unique():
    wmn_pred = pd.DataFrame({"year": years, "right": r})
    plt.plot(years, m1.predict(wmn_pred), label=f"{r} (H₁)")

# Common-slope model (H0)
for r in women_long["right"].unique():
    intercept = (
        m0.params["Intercept"]
        + m0.params.get(f"C(right)[T.{r}]", 0)
    )
    plt.plot(
        years,
        intercept + m0.params["year"] * years,
        color="grey",
        alpha=0.3
    )

plt.xlabel("Year")
plt.ylabel("% of countries with right")
plt.title("Observed heterogeneity vs common-slope counterfactual")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

Fig 5 (above)

The grey lines in Figure 5 represent counterfactual adoption trajectories implied by H0, under which all rights are assumed to diffuse at a common rate over time. Each grey line corresponds to a specific right and reflects a distinct baseline level of adoption (see the intercepts in the OLS charts). The lines are parallel, because H0 constrains all rights to share the same time slope, even though their starting points are different. This visual shows the core assumption of the common-slope model: that differences across rights are confined to baseline adoption levels rather than rates of diffusion.
A scatterplot was used during analysis but it was too difficult to interpret - there were many datapoints and it was very cluttered.
# In[77]:


#Set up for confidence intervals to test the hypothesis

params = m1.params
cov = m1.cov_params()

rights = sorted(women_long["right"].unique())

#Find all year and right interaction terms
interaction_terms = {}
for name in params.index:
    if name.startswith("year:C(right") and "[T." in name:
        # Extract the right label between [T. ... ]
        r = name.split("[T.", 1)[1].rstrip("]")
        interaction_terms[r] = name

#Baseline right (no interaction term)
baseline = [r for r in rights if r not in interaction_terms][0]
print("Baseline (reference) right:", baseline)

rows = []
for r in rights:
    base_slope = params["year"]
    base_var = cov.loc["year", "year"]

    if r == baseline:
        slope = base_slope
        var = base_var
    else:
        term = interaction_terms[r]              # use the real name
        slope = base_slope + params[term]
        var = base_var + cov.loc[term, term] + 2 * cov.loc["year", term]

    se = float(np.sqrt(var))
    rows.append({
        "right": r,
        "slope_pp_per_year": float(slope),
        "se": se,
        "ci_low": float(slope - 1.96 * se),
        "ci_high": float(slope + 1.96 * se)
    })

slopes_wmn = pd.DataFrame(rows).sort_values("slope_pp_per_year", ascending=False)

#Sanity check
print("Unique SEs:", slopes_wmn["se"].nunique())
slopes_wmn


# In[54]:


# Plot the CI graph
slopes_plot = slopes_wmn.sort_values("slope_pp_per_year")

plt.figure(figsize=(10, 5))  # wider so labels fit

plt.errorbar(
    slopes_plot["slope_pp_per_year"],
    slopes_plot["right"],
    xerr=[
        slopes_plot["slope_pp_per_year"] - slopes_plot["ci_low"],
        slopes_plot["ci_high"] - slopes_plot["slope_pp_per_year"]
    ],
    fmt="o",
    capsize=3
)

common_slope = m0.params["year"]
plt.axvline(common_slope, color="grey", linestyle="--", label=f"Common slope (H₀) = {common_slope:.2f}")

plt.xlabel("Change in % of countries per year")
plt.title("Estimated diffusion rates of women's rights (95% CI)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

Fig 6 (above)   The figure shows estimated annual diffusion rates for women’s rights, measured as the percentage-point increase per year in the share of countries adopting each right. Each point represents the estimated slope; horizontal bars indicate 95% confidence intervals. The vertical broken line marks the common diffusion rate implied by a model in which all rights share the same slope: non-overlap of the intervals with the null value indicates statistically significant differences in diffusion rates across all rights. Estimated slopes are clearly separated, with non-overlapping confidence intervals, indicating statistically distinct adoption rates across all rights. This supports the F-test and we can reject the null hypothesis.Lagged dynamics test - this tests whether early adoption predicts later adoption differently across rights.  Momentum of adoption, if it were the same across all rights, ewould indicate support for the H0.  Rights exhibiting stronger or weaker temporal persistence than others would indicate rejecting the H0.  
# In[78]:


#lagged dynamics test

women_long = women_long.sort_values(["entity", "right", "year"])

women_long["pct_lag5"] = (
    women_long
    .groupby(["entity", "right"])["pct_with_right"]
    .shift(5)
)
lagged_data = women_long.dropna(subset=["pct_lag5"])

m0_lag = smf.ols(
    "pct_with_right ~ pct_lag5 + C(right) + year",
    data=lagged_data
).fit()

m1_lag = smf.ols(
    "pct_with_right ~ pct_lag5 * C(right) + year",
    data=lagged_data
).fit()

F_stat, p_val, df_diff = m1_lag.compare_f_test(m0_lag)

print(f"F = {F_stat:.2f}, p = {p_val:.2e}, df = {df_diff}")

This F-test compares a lagged common-dynamics model to one allowing right-specific persistence. It rejects the null hypothesis of equal lagged effects (F = 4.14, p < 0.001). This indicates that earlier adoption levels predict subsequent diffusion differently across rights. An F-value of 4.14 is much smaller than the slope test, but is still non-trivial. Under the assumption that H0 represents the real world, the probability of observing an F-statistic as large as that obtained is very small.
# In[79]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assume lagged_data contains:
# pct_with_right, pct_lag5, right

plt.figure(figsize=(9,6))

sns.lmplot(
    data=lagged_data,
    x="pct_lag5",
    y="pct_with_right",
    hue="right",
    height=6,
    aspect=1.2,
    scatter=False,          # ← THIS LINE
    line_kws={"linewidth":2},
    ci=95
)

plt.xlabel("Adoption (%) five years earlier")
plt.ylabel("Current adoption (%)")
plt.title("Lagged diffusion dynamics by right")
plt.tight_layout()
plt.show()

Fig 7 (above) If adoption was x% five years ago, what do we expect adoption to be now?
The plot shows that higher past adoption is associated with higher current adoption for all rights, but the strength of this relationship varies across rights, indicating differences in diffusion persistence.  So diffusion of rights isn't governed by a single process, and different rights behave differently over time - ie differences in momentum. So we can reject the null hypothesis because lagged diffusion dynamics of each right are not shared.
A note on other possible hypotheses:
A hypothesis which tests whether there is a saturation point might be considered, since we can see that more and more countries adopt legislation over the time span, and there are only so many countries.  We could ask when saturation is likely to happen, given that all the graphs climb and then begin to level out, suggesting a slower rate of change, which would be what happens as fewer countries are left to adopt legislation. One of the graphs appears to have an S-shape (domestic violence legislation) but we can reject a quadratic model because there is considerable slope heterogeneity in the above visualisations of the data: too much to hypothesise a saturation point. Bias and misrepresentation:
This is not an experimental sample (the data are observed), so it can be suggested that there is a measurement bias.  The data represents how far and how fast countries have adopted rights legislation, but not whether it has produced any meaningful change for women.  In reference to other datasets, particularly the Sustainable Development Goals for women (Our World In Data Team, 2023), it can be seen that inequalities persist, and in much wider categories than those identified in the dataset under examination in this notebook.  At best, legislation may represent the will to strengthen lagal protections for women - in essence, a proxy bias - which under-represents the phenomenon of women's equality.
# In[80]:


women_long.to_csv("women_long_clean.csv", index=False)


# In[81]:


import os
os.getcwd()
os.listdir()

References:
Christopherson Puh, K. M., Yiadom, A., Johnson, J., Fernando, F., Yazid, H., & Thiemann, C. (2022). Tackling legal impediments to women’s economic empowerment (IMF Working Paper No. 2022/037). International Monetary Fund. https://doi.org/10.5089/9798400203640.001 

Our World in Data team (2023) - “Achieve gender equality and empower all women and girls” Published online at OurWorldinData.org. Retrieved from: 'https://archive.ourworldindata.org/20251209-133038/sdgs/gender-equality.html' [Online Resource] (archived on December 9, 2025).

Tertilt, M., Doepke, M., Hannusch, A. & Montenbruck, L. (2022). The economics of women’s rights: The Mary Paley and Alfred Marshall Lecture (CRC TR 224 Discussion Paper No. 372). University of Bonn & University of Mannheim. https://eprints.lse.ac.uk/117369/1/The_Economics_of_Women_s_Rights.pdf
