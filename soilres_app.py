import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import streamlit as st
import sympy as sp
from sympy import symbols
import io

st.set_page_config(
    page_title="Soil Resistivity Calculator",
    page_icon="logo.png",
    layout="wide"
)


st.title("Soil Resistivity Calculator")
#st.balloons()
#st.snow()

#======================
# Starting data- INPUT
#======================
with st.sidebar:
    st.title("Input Parameters")
    df = pd.DataFrame({
        "Pin Spacing (cm)": [50, 100, 150, 200, 250, 300, 400, 500],
        "Resistance (Ω)": [22.80, 11.56, 7.46, 5.66, 4.54, 3.78, 2.88, 2.29]
    })

    # Editable table
    edited_df = st.data_editor(
        df,
        hide_index=True,
        #num_rows="dynamic",
        use_container_width=True,
    )

#=====================
#Apparent Resistiity
#=====================
edited_df = edited_df.dropna(subset=["Pin Spacing (cm)", "Resistance (Ω)"])
def convert_to_matrix(input):
    '''convert to matrix(one column, n rows)'''
    matrix = np.array(input, dtype=float).reshape(-1,1)
    return(matrix)

pin_spacing = convert_to_matrix(edited_df["Pin Spacing (cm)"])
resistance = convert_to_matrix(edited_df["Resistance (Ω)"])

apparent_resistivity = 2 * (math.pi) * ((pin_spacing)/100) * (resistance)

lnSpacing = np.log(pin_spacing/100)
lnappres = np.log(apparent_resistivity)

def convert_to_1D_array(input):
    '''convert to 1D array for graph plotting'''
    oneDarray = np.array(input).reshape(-1)
    return(oneDarray)

array_ln_spacing = convert_to_1D_array(lnSpacing)
array_ln_appres = convert_to_1D_array(lnappres) 

#=======================================
# FINDING COEFFICIENTS FOR FITTED CURVE
#=======================================
st.divider()
st.header("Fitting Data to Curve")
st.write ("A cubic polynomial is fitted to ln(ρ_a) versus ln(a) using least-squares regression. The fit is evaluated on a 50-point logarithmic grid spanning the measured spacing range. This provides a smooth continuous curve for reading apparent resistivity at any spacing within the surveyed range. The fit is valid only within the data range and should not be extrapolated.")
coefficients = np.polyfit(array_ln_spacing, array_ln_appres, 3)    #predicts the cubic polynomial coefficients
A, B, C, D = coefficients
coeffsdata = pd.DataFrame({"Coefficient": ["a", "b", "c", "d"], "Value": [A, B, C, D]})

coeffstable = st.dataframe(
    coeffsdata,
    hide_index=True,
    use_container_width=True)

x, y, a, b, c, d = sp.symbols('x y a b c d')
y = a*x**3 + b*x**2 + c*x + d
Y = A*x**3 + B*x**2 + C*x + D
st.latex(r"y = " + sp.latex(y))
st.latex(r"y = " + sp.latex(Y))
#==============================
#FINDING DATA POINTS FOR GRAPHS
#==============================
grid_ln_spacing = np.linspace(
    np.min(array_ln_spacing),                   #plots the fitted data over 50 points
    np.max(array_ln_spacing),
        50
)
lnappres_pred = np.polyval(coefficients, grid_ln_spacing)      #predicts the apparent resistivity for each of the 50 spacing points
lnappres_fitted = np.polyval(coefficients, array_ln_spacing)   #predicts the apparent resistivity at each of the measured data points

appres_pred = np.exp(lnappres_pred)
grid_spacing = np.exp(grid_ln_spacing)

#===========================
#GOODNESS OF FIT
#===========================
st.divider()
st.header("Goodness of Fit Results")
def fSSres(actual, model):
    '''residual sum of squares, sum of sqared errors between actual values and model predictions'''
    SSres = np.sum((actual - model)**2)
    return(SSres)
def fSStot(actual):
    '''total sum of squares- total variance in the actual data measured against the mean'''
    SStot = np.sum((actual - np.mean(actual))**2)
    return SStot
def fR_sq(SSres, SStot):
    '''R**2 = 1 - (SSres/SStot)'''
    R_sq = 1 - (SSres/SStot)
    return(R_sq)
def fRMSE(actual, model):
    '''square root of the average squared differences between a model's predicted values and the actual observed values'''
    no_of_coefficients = 4
    rmse = math.sqrt(fSSres(actual, model)/(len(actual)-no_of_coefficients))
    return(rmse)

SSres = fSSres(convert_to_matrix(array_ln_appres), convert_to_matrix(lnappres_fitted))    #calculating residual sum of squares
SStot = fSStot(convert_to_matrix(array_ln_appres))   #calculating total sum of squares
R_sq = fR_sq(SSres, SStot)
rmse = fRMSE(array_ln_appres, lnappres_fitted)

left, right = st.columns([1,1])
with left:
    fitdata = pd.DataFrame({"Parameter": ["SSres", "SStot", "Rsq", "rmse"], "Value": [SSres, SStot, R_sq, rmse]})
    fitstable = st.dataframe(
        fitdata,
        hide_index=True,
        use_container_width=True)
    with st.expander("Description of above terms"):
        st.write(
    "Goodness-of-Fit Assessment — Cubic Polynomial Regression in Log-Log Space\n\n"
    "The regression is performed on the log-transformed variables "
    "ln(ρₐ) versus ln(a), consistent with the power-law character of "
    "apparent resistivity variation with electrode spacing. The goodness-of-fit "
    "metrics are therefore evaluated in log-log space, where the residuals are "
    "physically meaningful. Computing R² in linear space would give a misleadingly "
    "optimistic result for monotonically trending data and is not used here.\n\n"
    
    "R² is the coefficient of determination, calculated from the residual and "
    "total sums of squares of ln(ρₐ). It quantifies the proportion of variance "
    "in the log-transformed apparent resistivity that is explained by the cubic "
    "polynomial model. A threshold of R² > 0.995 is applied; datasets falling "
    "below this threshold may indicate a measurement outlier, electrode contact "
    "issues, or a soil structure that is not well represented by a smooth "
    "monotonic profile and warrants engineering review.\n\n"
    
    "RMSE is the root mean square error of the residuals in log space. It is "
    "converted to a multiplicative fit factor via exp(RMSE), which gives the "
    "typical ratio between the fitted and measured apparent resistivity values "
    "in linear space. This is a more physically interpretable metric than the "
    "log-space RMSE alone. A fit factor below 1.05 corresponds to a typical "
    "point error of less than 5% in apparent resistivity, which is well within "
    "the measurement uncertainty of the Wenner method in field conditions.\n\n"
    
    "Note that R² and fit factor are not independent — both are derived from "
    "the same residuals — but they convey complementary information: R² assesses "
    "overall model adequacy relative to data variance, while fit factor gives "
    "an absolute sense of point-by-point error in engineering units."
)

with right:
        with st.container(border=True, vertical_alignment="center"):
            st.subheader("Fit Factor", text_alignment="center")
            fit_factor =  math.exp(fRMSE(array_ln_appres, lnappres_fitted))
            st.write(f"\nFit Factor: {fit_factor:.4f}\n")      #calculating fit factor
            if fit_factor<1.1:
                st.badge("Fit quality: Pass✅", color='green')
            else:
                st.badge("Fit quality: Fail❌", color='red')
            st.space("xsmall")
        with st.expander("Click to see description of Fit Factor"):
            st.write("How well does the curve fit the measurements? \n\nThe smooth curve drawn through the data points is calculated automatically by the worksheet using a mathematical fitting method. It does not pass exactly through every measured point — instead it finds the best overall smooth curve through all of the points taken together.\n\nTo check how well the curve fits, the worksheet calculates two numbers: Rsq is a measure of how closely the curve follows the measured data points. A value of 1.000 would mean a perfect fit. In practice a value above 0.995 indicates the curve is a very good representation of the measurements and can be used with confidence. If Rsq is below 0.995, the fit should be reviewed — there may be an unusual measurement in the dataset or the soil profile may be more complex than the curve can represent.\n\nFit Factor gives a practical sense of the typical error between the curve and the individual measurements. A fit factor of 1.03 means that at any given pin spacing, the curve value is typically within 3% of the measured value. Values below 1.05 (i.e. within 5%) are generally acceptable for soil resistivity work.\n\nIf the worksheet shows PASS, the curve fit is acceptable and the results can be used. If it shows FAIL, the data needs to be reviewed more carefully.")

#===========================
#MINIMUMS AND MAXIMUMS
#===========================
st.divider()
st.header("Minimum and Maximum Values")
min_spacing = min(array_ln_spacing)
fit_res_at_min_spacing = np.exp(np.polyval(coefficients, min_spacing))

max_spacing = max(array_ln_spacing)
fit_res_at_max_spacing = np.exp(np.polyval(coefficients, max_spacing))

min_max_data = pd.DataFrame({"Parameters": ["Fitted Resistivity at Minimum Spacing",
                                            "Fitted Resistivity at Maximum Spacing",
                                            "Measured minimum value",
                                            "Measured Maximum Value"],
                                            "Values": [fit_res_at_min_spacing,
                                                       fit_res_at_max_spacing,
                                                      float(np.min(apparent_resistivity)),
                                                       float(np.max(apparent_resistivity))],
                                                       "Units": ["Ωm", "Ωm", "Ωm", "Ωm" ]})
man_max_table = st.dataframe(
        min_max_data,
        hide_index=True,
        use_container_width=True)

#=========================
#PLOTTING DATA
#=========================
st.header("Data Plot")
fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(
    np.exp(array_ln_spacing),
    np.exp(array_ln_appres),
    label="Measured data",
    zorder=2
)

ax.plot(
    grid_spacing,
    appres_pred,
    color="red",
    linewidth=2,
    label="Cubic fit",
    zorder=1
)

ax.set_title("Apparent Resistivity vs Pin Spacing")
ax.set_xlabel("Spacing (m)")
ax.set_ylabel("Apparent Resistivity (Ωm)")

ax.legend()
ax.grid(which="major", color="gray", linestyle="-", linewidth=0.7, alpha=0.6)

ax.minorticks_on()
ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

st.pyplot(fig)

#===================
#EXPORTING TO EXCEL
#===================
# -----------------------------
# Prepare data for Excel export
# -----------------------------

# Measured data
measured_df = pd.DataFrame({
    "Pin Spacing (cm)": edited_df["Pin Spacing (cm)"].to_numpy(),
    "Pin Spacing (m)": (
        edited_df["Pin Spacing (cm)"] / 100
    ).to_numpy(),
    "Resistance (Ω)": edited_df["Resistance (Ω)"].to_numpy(),
    "Apparent Resistivity (Ωm)": apparent_resistivity.flatten(),
    "Fitted Resistivity (Ωm)": np.exp(lnappres_fitted).flatten()
})

# Cubic fitted curve data
fitted_curve_df = pd.DataFrame({
    "Spacing (m)": grid_spacing,
    "Fitted Resistivity (Ωm)": appres_pred
})

# -----------------------------
# Create Excel file
# -----------------------------

excel_buffer = io.BytesIO()

with pd.ExcelWriter(
    excel_buffer,
    engine="openpyxl"
) as writer:

    measured_df.to_excel(
        writer,
        index=False,
        sheet_name="Measured Data"
    )

    fitted_curve_df.to_excel(
        writer,
        index=False,
        sheet_name="Cubic Fit"
    )

excel_buffer.seek(0)

# -----------------------------
# Download button
# -----------------------------

st.download_button(
    label="📥 Export Graph Data to Excel",
    data=excel_buffer,
    file_name="apparent_resistivity_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)