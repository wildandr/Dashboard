import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pickle
from scipy.optimize import minimize


# # Memuat model yang disimpan dengan pickle
# with open('ridge_model.pkl', 'rb') as f:
#     ridge_model = pickle.load(f)

# # Memuat scaler yang digunakan untuk normalisasi input dan output
# with open('scaler_x.pkl', 'rb') as f:
#     scaler_x = pickle.load(f)
    
# with open('scaler_y.pkl', 'rb') as f:
#     scaler_y = pickle.load(f)

# Kolom Input dari pengguna
input_ni = st.number_input("Masukkan Ni:", min_value=0.0, format="%.10f", value=2.055)
input_fe = st.number_input("Masukkan Fe:", min_value=0.0, format="%.10f", value=13.265)
input_sio2 = st.number_input("Masukkan SiO2:", min_value=0.0, format="%.10f", value=39.84)
input_cao = st.number_input("Masukkan CaO:", min_value=0.0, format="%.10f", value=0.46)
input_mgo = st.number_input("Masukkan MgO:", min_value=0.0, format="%.10f", value=23.195)
input_al2o3 = st.number_input("Masukkan Al2O3:", min_value=0.0, format="%.10f", value=1.8)
input_fe_ni = st.number_input("Masukkan Fe/Ni Ratio:", min_value=0.0, format="%.10f", value=6.451668721)
input_s_m = st.number_input("Masukkan S/M Ratio:", min_value=0.0, format="%.10f", value=1.718814135)
input_bc = st.number_input("Masukkan BC:", min_value=0.0, format="%.10f", value=0.590599279)
input_loi_in = st.number_input("Masukkan LOI_in:", min_value=0.0, format="%.10f", value=13.195)
input_mc_kilnfeed = st.number_input("Masukkan MC Kilnfeed:", min_value=0.0, format="%.10f", value=20.955)
input_fc_coal = st.number_input("Masukkan FC Coal:", min_value=0.0, format="%.10f", value=44.71)
input_gcv_coal = st.number_input("Masukkan GCV Coal:", min_value=0.0, format="%.10f", value=658.97)
input_tco = st.number_input("Masukkan TCO:", min_value=0.0, format="%.10f", value=144.21)
input_charge_klin = st.number_input("Masukkan Charge Klin:", min_value=0.0, format="%.10f", value=2242.24)

# Tombol untuk mengirim data
if st.button("Kirim"):
    df_1= pd.read_excel(r'Merge Table to_RK and to_EF part 2 (1).xlsx', sheet_name='shifting_7-7_16', skiprows=0)
    df = df_1.drop(columns=['date', 'time']).copy()
    df = df.drop(index=range(0, 16)).reset_index(drop=True)
    df = df.iloc[:-7].reset_index(drop=True)
    input_cols = df.loc[:, 'ni_in':'t_tic172'].columns
    output_cols = df.loc[:, 'furnace_temp':'loi_kalsin'].columns
    df[input_cols.union(output_cols)] = df[input_cols.union(output_cols)].interpolate(method='linear', axis=0)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_norm = pd.DataFrame(scaler_x.fit_transform(df[input_cols]), columns=input_cols)
    y_norm = pd.DataFrame(scaler_y.fit_transform(df[output_cols]), columns=output_cols)
    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

    ridge_model = Ridge(alpha=0.1, solver='svd')  # Regularization strength

    # Train the model
    ridge_model.fit(X_train.to_numpy(), y_train)

    # Predict on test set
    y_pred_rr = ridge_model.predict(X_test.to_numpy())
    y_pred_rr = scaler_y.inverse_transform(y_pred_rr)
    y_pred_rr_df = pd.DataFrame(y_pred_rr, columns=output_cols)
    X_test_inverted = scaler_x.inverse_transform(X_test)
    X_test_inverted = pd.DataFrame(X_test_inverted, columns=input_cols)
    inverted_pred_rr_df = pd.concat([X_test_inverted.reset_index(drop=True), y_pred_rr_df.reset_index(drop=True)], axis=1)

    inverted_y_actual = scaler_y.inverse_transform(y_test)
    inverted_y_actual = pd.DataFrame(inverted_y_actual, columns=output_cols)
    inverted_y_pred_rr = y_pred_rr_df[output_cols]

    models = {
        'RR': inverted_y_pred_rr,

    }
    mae_dict = {'Column': output_cols}
    # Iterate through each model and calculate MAE for each column
    for model_name, inverted_y_pred in models.items():
        mae_dict[model_name] = []  # Add a column for each model
        for col in output_cols:
            # Extract actual and predicted values
            y_actual_col = inverted_y_actual[col]
            y_pred_col = inverted_y_pred[col]

            # Calculate MAE and store in the dictionary
            mae_col = mean_absolute_error(y_actual_col, y_pred_col)
            mae_dict[model_name].append(mae_col)

    # Convert the dictionary to a DataFrame
    metrics_df = pd.DataFrame(mae_dict)

    # Memuat model yang disimpan dengan pickle
    with open('ridge_model.pkl', 'rb') as f:
        ridge_model = pickle.load(f)

    # Memuat scaler_x dari file
    with open('scaler_x.pkl', 'rb') as f:
        scaler_x = pickle.load(f)

    # Memuat scaler_y dari file
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    target_value = y_test.iloc[0, :].to_numpy()

    def objective_function(x_scaled):
        # untuk  prediksi 1 fitur output saja dengan i sample saja pada 30 fitur input
        x_scaled=x_scaled.reshape(1,-1)
        y_pred = ridge_model.predict(x_scaled)[0]
        return (y_pred[0] - target_value[0]) ** 2

    min_values = scaler_y.data_min_
    max_values = scaler_y.data_max_

    konstrain=[]
    bounds=[ (i,j) for i, j in zip(min_values, max_values)]

    T_furnace=1300
    T_furnace_norm= (T_furnace-bounds[0][0])/(bounds[0][1]-bounds[1][0])

    ni_met=17
    ni_met_norm= (ni_met-bounds[1][0])/(bounds[1][1]-bounds[1][0])

    C_met_low=0
    C_met_low_norm= (C_met_low-bounds[2][0])/(bounds[2][1]-bounds[2][0])

    Si_met_low=1
    Si_met_low_norm= (Si_met_low-bounds[3][0])/(bounds[3][1]-bounds[3][0])

    Si_met_high=2
    Si_met_high_norm= (Si_met_high-bounds[3][0])/(bounds[3][1]-bounds[3][0])

    fe_met_low=0
    fe_met_low_norm= (fe_met_low-bounds[4][0])/(bounds[4][1]-bounds[4][0])

    s_met_low=0
    s_met_low_norm= (s_met_low-bounds[5][0])/(bounds[5][1]-bounds[5][0])

    s_met_high=0.4
    s_met_high_norm= (s_met_high-bounds[5][0])/(bounds[5][1]-bounds[5][0])

    ni_slag_low=0
    ni_slag_low_norm= (ni_slag_low-bounds[6][0])/(bounds[6][1]-bounds[6][0])

    fe_slag_low=0
    fe_slag_low_norm= (fe_slag_low-bounds[7][0])/(bounds[7][1]-bounds[7][0])

    T_kalsin_low=600
    T_kalsin_low_norm= (T_kalsin_low-bounds[8][0])/(bounds[8][1]-bounds[8][0])

    pic_161_low=-1
    pic_161_low_norm= (pic_161_low-bounds[9][0])/(bounds[9][1]-bounds[9][0])

    loi_kalsin_low=0
    loi_kalsin_low_norm= (loi_kalsin_low-bounds[10][0])/(bounds[10][1]-bounds[10][0])

    loi_kalsin_high=1
    loi_kalsin_high_norm= (loi_kalsin_high-bounds[10][0])/(bounds[10][1]-bounds[10][0])

    x0 = X_test.iloc[0,:].to_numpy().flatten()
    min_values_input = scaler_x.data_min_
    max_values_input = scaler_x.data_max_

    konstrain_input=[ (i,j) for i, j in zip(min_values_input, max_values_input)]

    ni_in_min=0
    ni_in_min_norm= (ni_in_min-konstrain_input[0][0])/(konstrain_input[0][1]-konstrain_input[0][0])

    fe_in_min=0
    fe_in_min_norm= (fe_in_min-konstrain_input[1][0])/(konstrain_input[1][1]-konstrain_input[1][0])

    sio2_in_min=0
    sio2_in_min_norm= (sio2_in_min-konstrain_input[2][0])/(konstrain_input[2][1]-konstrain_input[2][0])

    cao_in_min=0
    cao_in_min_norm= (cao_in_min-konstrain_input[3][0])/(konstrain_input[3][1]-konstrain_input[3][0])

    mgo_in_min=0
    mgo_in_min_norm= (mgo_in_min-konstrain_input[4][0])/(konstrain_input[4][1]-konstrain_input[4][0])

    al2o3_in_min=0
    al2o3_in_min_norm= (al2o3_in_min-konstrain_input[5][0])/(konstrain_input[5][1]-konstrain_input[5][0])

    fe_ni_min=0
    fe_ni_min_norm= (fe_ni_min-konstrain_input[6][0])/(konstrain_input[6][1]-konstrain_input[6][0])

    s_m_min=0
    s_m_min_norm= (s_m_min-konstrain_input[7][0])/(konstrain_input[7][1]-konstrain_input[7][0])

    bc_min=0
    bc_min_norm= (bc_min-konstrain_input[8][0])/(konstrain_input[8][1]-konstrain_input[8][0])

    loi_in_min=0
    loi_in_min_norm= (loi_in_min-konstrain_input[9][0])/(konstrain_input[9][1]-konstrain_input[9][0])

    mc_kilnfeed_exact_norm=x0[10]
    fc_coal_exact_norm=x0[11]
    gvc_coal_exact_norm=x0[12]
    tco_exact_norm=x0[13]

    voltage_min=0
    voltage_min_norm= (voltage_min-konstrain_input[14][0])/(konstrain_input[14][1]-konstrain_input[14][0])

    current_min=0
    current_min_norm= (current_min-konstrain_input[15][0])/(konstrain_input[15][1]-konstrain_input[15][0])

    load_min=0
    load_min_norm= (load_min-konstrain_input[16][0])/(konstrain_input[16][1]-konstrain_input[16][0])

    rpm_min=0
    rpm_min_norm= (rpm_min-konstrain_input[17][0])/(konstrain_input[17][1]-konstrain_input[17][0])

    pry_p_min=0
    pry_p_min_norm= (pry_p_min-konstrain_input[18][0])/(konstrain_input[18][1]-konstrain_input[18][0])

    sec_p_min=0
    sec_p_min_norm= (sec_p_min-konstrain_input[19][0])/(konstrain_input[19][1]-konstrain_input[19][0])

    pry_v_min=0
    pry_v_min_norm= (pry_v_min-konstrain_input[20][0])/(konstrain_input[20][1]-konstrain_input[20][0])

    sec_v_min=0
    sec_v_min_norm= (sec_v_min-konstrain_input[21][0])/(konstrain_input[21][1]-konstrain_input[21][0])

    total_coal_min=0
    total_coal_min_norm= (total_coal_min-konstrain_input[22][0])/(konstrain_input[22][1]-konstrain_input[22][0])

    a_f_ratio_min=0
    a_f_ratio_min_norm= (a_f_ratio_min-konstrain_input[23][0])/(konstrain_input[23][1]-konstrain_input[23][0])

    kg_tco_exact_norm=x0[24]

    reductor_ratio_min=0
    reductor_ratio_min_norm= (reductor_ratio_min-konstrain_input[25][0])/(konstrain_input[25][1]-konstrain_input[25][0])

    reductor_consume_min=0
    reductor_consume_min_norm= (reductor_consume_min-konstrain_input[26][0])/(konstrain_input[26][1]-konstrain_input[26][0])

    charge_kiln_exact_norm=x0[27]

    t_tic162_min=556
    t_tic162_min_norm= (t_tic162_min-konstrain_input[28][0])/(konstrain_input[28][1]-konstrain_input[28][0])

    t_tic162_max=1201
    t_tic162_max_norm= (t_tic162_max-konstrain_input[28][0])/(konstrain_input[28][1]-konstrain_input[28][0])

    t_tic163_min=456
    t_tic163_min_norm= (t_tic163_min-konstrain_input[29][0])/(konstrain_input[29][1]-konstrain_input[29][0])

    t_tic163_max=845
    t_tic163_max_norm= (t_tic163_max-konstrain_input[29][0])/(konstrain_input[30][1]-konstrain_input[30][0])

    t_tic166_min=372
    t_tic166_min_norm= (t_tic166_min-konstrain_input[30][0])/(konstrain_input[30][1]-konstrain_input[30][0])

    t_tic166_max=867
    t_tic166_max_norm= (t_tic166_max-konstrain_input[30][0])/(konstrain_input[30][1]-konstrain_input[30][0])

    t_tic172_min=210
    t_tic172_min_norm= (t_tic172_min-konstrain_input[31][0])/(konstrain_input[31][1]-konstrain_input[31][0])

    t_tic172_max=315
    t_tic172_max_norm= (t_tic172_max-konstrain_input[31][0])/(konstrain_input[31][1]-konstrain_input[31][0])

    constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0] - ni_in_min_norm},  # ni_in > 0
            {'type': 'ineq', 'fun': lambda x: x[1] - fe_in_min_norm},  # fe_in > 0
            {'type': 'ineq', 'fun': lambda x: x[2] - sio2_in_min_norm},  # sio2_in > 0
            {'type': 'ineq', 'fun': lambda x: x[3] - cao_in_min_norm},  # cao_in > 0
            {'type': 'ineq', 'fun': lambda x: x[4] - mgo_in_min_norm},  # mgo_in > 0
            {'type': 'ineq', 'fun': lambda x: x[5] - al2o3_in_min_norm},  # al2o3_in > 0
            {'type': 'ineq', 'fun': lambda x: x[6] - fe_ni_min_norm},  # fe_ni > 0
            {'type': 'ineq', 'fun': lambda x: x[7] - s_m_min_norm},  # s_m > 0
            {'type': 'ineq', 'fun': lambda x: x[8] - bc_min_norm},  # bc > 0
            {'type': 'ineq', 'fun': lambda x: x[9] - loi_in_min_norm},  # loi_in > 0

            {'type': 'eq', 'fun': lambda x: mc_kilnfeed_exact_norm - x[10]},  # mc_kilnfeed tidak berubah
            {'type': 'eq', 'fun': lambda x: fc_coal_exact_norm - x[11]},  # fc_coal tidak berubah
            {'type': 'eq', 'fun': lambda x: gvc_coal_exact_norm - x[12]},  # gvc_coal tidak berubah
            {'type': 'eq', 'fun': lambda x: tco_exact_norm - x[13]},  # tco tidak berubah

            {'type': 'ineq', 'fun': lambda x: x[14] - voltage_min_norm},  # voltage > 0
            {'type': 'ineq', 'fun': lambda x: x[15] - current_min_norm},  # current > 0
            {'type': 'ineq', 'fun': lambda x: x[16] - load_min_norm},  # load > 0
            {'type': 'ineq', 'fun': lambda x: x[17] - rpm_min_norm},  # rpm > 0
            {'type': 'ineq', 'fun': lambda x: x[18] - pry_p_min_norm},  # pry_p > 0
            {'type': 'ineq', 'fun': lambda x: x[19] - sec_p_min_norm},  # sec_p > 0
            {'type': 'ineq', 'fun': lambda x: x[20] - pry_v_min_norm},  # pry_v > 0
            {'type': 'ineq', 'fun': lambda x: x[21] - sec_v_min_norm},  # sec_v > 0
            {'type': 'ineq', 'fun': lambda x: x[22] - total_coal_min_norm},  # total_coal > 0
            {'type': 'ineq', 'fun': lambda x: x[23] - a_f_ratio_min_norm},  # a_f_ratio > 0

            {'type': 'eq', 'fun': lambda x: kg_tco_exact_norm - x[24]},  # kg_tco tidak berubah

            {'type': 'ineq', 'fun': lambda x: x[25] - reductor_ratio_min_norm},  # reductor_ratio > 0
            {'type': 'ineq', 'fun': lambda x: x[26] - reductor_consume_min_norm},  # reductor_consume > 0

            {'type': 'eq', 'fun': lambda x: charge_kiln_exact_norm - x[27]},  # charge_kiln tidak berubah

            {'type': 'ineq', 'fun': lambda x: x[28] - t_tic162_min_norm},  # t_tic162 > 556
            {'type': 'ineq', 'fun': lambda x: t_tic162_max_norm - x[28]},  # t_tic162 < 1201
            {'type': 'ineq', 'fun': lambda x: x[29] - t_tic163_min_norm},  # t_tic163 > 456
            {'type': 'ineq', 'fun': lambda x: t_tic163_max_norm - x[29]},  # t_tic163 < 845
            {'type': 'ineq', 'fun': lambda x: x[30] - t_tic166_min_norm},  # t_tic166 > 372
            {'type': 'ineq', 'fun': lambda x: t_tic166_max_norm - x[30]},  # t_tic166 < 867
            {'type': 'ineq', 'fun': lambda x: x[31] - t_tic172_min_norm},  # t_tic172 > 210
            {'type': 'ineq', 'fun': lambda x: t_tic172_max_norm - x[31]},  # t_tic172 < 315

        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][0] - T_furnace_norm},  # T_furnace > 1300
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][1] - ni_met_norm},    # Ni_met > 17
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][2] - C_met_low_norm},  # C_met > 0
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][3] - Si_met_low_norm},  # Si_met > 1
        {'type': 'ineq', 'fun': lambda x: Si_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][3]},  # Si_met < 2
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][4] - fe_met_low_norm},  # Fe_met > 0
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][5] - s_met_low_norm},  # S_met > 0
        {'type': 'ineq', 'fun': lambda x: s_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][5]},  # S_met < 0.4
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][6] - ni_slag_low_norm},  # Ni_slag > 0
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][7] - fe_slag_low_norm},  # Fe_slag > 0
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][8] -  T_kalsin_low_norm},    # T_kalsin > 600
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][9] - pic_161_low_norm},  # pic_161 > -3
        {'type': 'ineq', 'fun': lambda x: loi_kalsin_high - ridge_model.predict(x.reshape(1, -1))[0][10]},  # loi_kalsin < 1
        {'type': 'ineq', 'fun': lambda x: ridge_model.predict(x.reshape(1, -1))[0][10] - loi_kalsin_low },  # loi_kalsin > 0
    ]

    # Jalankan optimisasi dengan constraints
    result = minimize(objective_function, x0, method='SLSQP', constraints=constraints)

    optimized_x = result.x
    optimized_y = ridge_model.predict(optimized_x.reshape(1,-1))[0]

    # Inversi hasil optimasi x (untuk input)
    x_optimized_inversed = scaler_x.inverse_transform([optimized_x])[0]

    # Inversi hasil optimasi y (untuk output)
    y_optimized_inversed = scaler_y.inverse_transform([optimized_y])[0]

    a=pd.DataFrame(optimized_x).T
    a.columns=input_cols
    a=pd.DataFrame(scaler_x.inverse_transform(a), columns=input_cols)
    b=pd.DataFrame(optimized_y).T
    b=pd.DataFrame(scaler_y.inverse_transform(b),columns=output_cols)
    y_test_inverted = scaler_y.inverse_transform(y_test)
    y_test_inverted = pd.DataFrame(y_test_inverted, columns=output_cols)

    y_pred = ridge_model.predict(X_test.iloc[3, :].to_numpy().flatten().reshape(1, -1)).flatten()
    y_pred = ridge_model.predict(X_test.iloc[116, :].to_numpy().reshape(1, -1))

    # Input pengguna (15 nilai yang tidak dioptimalkan, dikunci)
    user_input = {
        'ni_in': input_ni,
        'fe_in': input_fe,
        'sio2_in': input_sio2,
        'cao_in': input_cao,
        'mgo_in': input_mgo,
        'al2o3_in': input_al2o3,
        'fe_ni': input_fe_ni,
        's_m': input_s_m,
        'bc': input_bc,
        'loi_in': input_loi_in,
        'mc_kilnfeed': input_mc_kilnfeed,
        'fc_coal': input_fc_coal,
        'gcv_coal': input_gcv_coal,
        'tco': input_tco,
        'charge_kiln': input_charge_klin,
    }

    # Input untuk 17 nilai yang akan dinormalisasi secara acak
    random_input = np.random.uniform(low=0, high=1, size=(32 - len(user_input)))

    # Urutan input yang lengkap
    input_order = [
        'ni_in', 'fe_in', 'sio2_in', 'cao_in', 'mgo_in', 'al2o3_in', 'fe_ni', 's_m', 'bc', 'loi_in',
        'mc_kilnfeed', 'fc_coal', 'gcv_coal', 'tco', 'voltage', 'current', 'load', 'rpm', 'pry_p', 'sec_p',
        'pry_v', 'sec_v', 'total_coal', 'a_f_ratio', 'kg_tco', 'reductor_ratio', 'reductor_consume', 'charge_kiln',
        't_tic162', 't_tic163', 't_tic166', 't_tic172'
    ]

    # Menyusun nilai input sesuai urutan yang benar
    full_input = []
    for col in input_order:
        if col in user_input:
            # Menambahkan input pengguna yang dikunci
            full_input.append(user_input[col])
        else:
            # Menambahkan nilai acak untuk input yang tidak diberikan oleh pengguna
            full_input.append(random_input[0])
            random_input = random_input[1:]  # Mengurangi array random_input

    # Mengonversi full_input ke dalam array numpy
    full_input = np.array(full_input)

    # Menormalisasi input menggunakan scaler yang sudah disimpan
    full_input_scaled = scaler_x.transform(full_input.reshape(1, -1))

    # Mengakses nilai-nilai khusus dari array yang dinormalisasi
    ni_in_exact_norm = full_input_scaled[0][input_order.index('ni_in')]
    fe_in_exact_norm = full_input_scaled[0][input_order.index('fe_in')]
    sio2_in_exact_norm = full_input_scaled[0][input_order.index('sio2_in')]
    cao_in_exact_norm = full_input_scaled[0][input_order.index('cao_in')]
    mgo_in_exact_norm = full_input_scaled[0][input_order.index('mgo_in')]
    al2o3_in_exact_norm = full_input_scaled[0][input_order.index('al2o3_in')]
    fe_ni_exact_norm = full_input_scaled[0][input_order.index('fe_ni')]
    s_m_exact_norm = full_input_scaled[0][input_order.index('s_m')]
    bc_exact_norm = full_input_scaled[0][input_order.index('bc')]
    loi_in_exact_norm = full_input_scaled[0][input_order.index('loi_in')]
    mc_kilnfeed_exact_norm = full_input_scaled[0][input_order.index('mc_kilnfeed')]
    fc_coal_exact_norm = full_input_scaled[0][input_order.index('fc_coal')]
    gvc_coal_exact_norm = full_input_scaled[0][input_order.index('gcv_coal')]
    tco_exact_norm = full_input_scaled[0][input_order.index('tco')]
    kg_tco_exact_norm = full_input_scaled[0][input_order.index('kg_tco')]

    optimized_x_list = []
    optimized_y_list = []

    # Melakukan iterasi melalui 10 sampel pertama di X_test
    for i in range(1):
        print(f"\nMemproses Sampel ke-{i+1}")
        # Mengambil sampel ke-i dan meratakan array-nya
        x0 = X_test.iloc[i, :].to_numpy().flatten()

        # Pastikan y_test hanya memiliki satu kolom output jika prediksi satu fitur
        target_value = y_test.iloc[i, :].to_numpy()

        # Mendefinisikan fungsi objektif spesifik untuk sampel ke-i
        def objective_function(x_scaled):
            # Mengubah bentuk input untuk prediksi
            x_scaled_reshaped = x_scaled.reshape(1, -1)
            # Melakukan prediksi menggunakan model ridge
            y_pred = ridge_model.predict(x_scaled_reshaped)[0]
            # Menghitung selisih kuadrat (MSE)
            return (y_pred[i] - target_value[i]) ** 2

        # Mendefinisikan constraints spesifik untuk sampel ke-i
        constraints_sample = [
            # Constraints equality (tidak berubah) (harusnya equality constraint, dikunci berdasarkan data asli)
            {'type': 'eq', 'fun': lambda x, ni_in_exact_norm=ni_in_exact_norm: x[0] - ni_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, fe_in_exact_norm=fe_in_exact_norm: x[1] - fe_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, sio2_in_exact_norm=sio2_in_exact_norm: x[2] - sio2_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, cao_in_exact_norm=cao_in_exact_norm: x[3] - cao_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, mgo_in_exact_norm=mgo_in_exact_norm: x[4] - mgo_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, al2o3_in_exact_norm=al2o3_in_exact_norm: x[5] - al2o3_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, fe_ni_exact_norm=fe_ni_exact_norm: x[6] - fe_ni_exact_norm},
            {'type': 'eq', 'fun': lambda x, s_m_exact_norm=s_m_exact_norm: x[7] - s_m_exact_norm},
            {'type': 'eq', 'fun': lambda x, bc_exact_norm=bc_exact_norm: x[8] - bc_exact_norm},
            {'type': 'eq', 'fun': lambda x, loi_in_exact_norm=loi_in_exact_norm: x[9] - loi_in_exact_norm},

            # Constraints equality (tidak berubah)
            {'type': 'eq', 'fun': lambda x, mc_kilnfeed_exact_norm=mc_kilnfeed_exact_norm: mc_kilnfeed_exact_norm - x[10]},  # mc_kilnfeed tidak berubah
            {'type': 'eq', 'fun': lambda x, fc_coal_exact_norm=fc_coal_exact_norm: fc_coal_exact_norm - x[11]},  # fc_coal tidak berubah
            {'type': 'eq', 'fun': lambda x, gvc_coal_exact_norm=gvc_coal_exact_norm: gvc_coal_exact_norm - x[12]},  # gvc_coal tidak berubah
            {'type': 'eq', 'fun': lambda x, tco_exact_norm=tco_exact_norm: tco_exact_norm - x[13]},  # tco tidak berubah

            # Constraints input >= min_norm
            {'type': 'ineq', 'fun': lambda x, voltage_min_norm=voltage_min_norm: x[14] - voltage_min_norm},  # voltage > voltage_min_norm
            {'type': 'ineq', 'fun': lambda x, current_min_norm=current_min_norm: x[15] - current_min_norm},  # current > current_min_norm
            {'type': 'ineq', 'fun': lambda x, load_min_norm=load_min_norm: x[16] - load_min_norm},  # load > load_min_norm
            {'type': 'ineq', 'fun': lambda x, rpm_min_norm=rpm_min_norm: x[17] - rpm_min_norm},  # rpm > rpm_min_norm
            {'type': 'ineq', 'fun': lambda x, pry_p_min_norm=pry_p_min_norm: x[18] - pry_p_min_norm},  # pry_p > pry_p_min_norm
            {'type': 'ineq', 'fun': lambda x, sec_p_min_norm=sec_p_min_norm: x[19] - sec_p_min_norm},  # sec_p > sec_p_min_norm
            {'type': 'ineq', 'fun': lambda x, pry_v_min_norm=pry_v_min_norm: x[20] - pry_v_min_norm},  # pry_v > pry_v_min_norm
            {'type': 'ineq', 'fun': lambda x, sec_v_min_norm=sec_v_min_norm: x[21] - sec_v_min_norm},  # sec_v > sec_v_min_norm
            {'type': 'ineq', 'fun': lambda x, total_coal_min_norm=total_coal_min_norm: x[22] - total_coal_min_norm},  # total_coal > total_coal_min_norm
            {'type': 'ineq', 'fun': lambda x, a_f_ratio_min_norm=a_f_ratio_min_norm: x[23] - a_f_ratio_min_norm},  # a_f_ratio > a_f_ratio_min_norm

            # Constraints equality (tidak berubah)
            {'type': 'eq', 'fun': lambda x, kg_tco_exact_norm=kg_tco_exact_norm: kg_tco_exact_norm - x[24]},  # kg_tco tidak berubah

            # Constraints input >= min_norm
            {'type': 'ineq', 'fun': lambda x, reductor_ratio_min_norm=reductor_ratio_min_norm: x[25] - reductor_ratio_min_norm},  # reductor_ratio > reductor_ratio_min_norm
            {'type': 'ineq', 'fun': lambda x, reductor_consume_min_norm=reductor_consume_min_norm: x[26] - reductor_consume_min_norm},  # reductor_consume > reductor_consume_min_norm

            # Constraints equality (tidak berubah)
            {'type': 'eq', 'fun': lambda x, charge_kiln_exact_norm=charge_kiln_exact_norm: charge_kiln_exact_norm - x[27]},  # charge_kiln tidak berubah

            # Constraints input >= min_norm dan <= max_norm
            {'type': 'ineq', 'fun': lambda x, t_tic162_min_norm=t_tic162_min_norm: x[28] - t_tic162_min_norm},  # t_tic162 > t_tic162_min_norm
            {'type': 'ineq', 'fun': lambda x, t_tic162_max_norm=t_tic162_max_norm: t_tic162_max_norm - x[28]},  # t_tic162 < t_tic162_max_norm
            {'type': 'ineq', 'fun': lambda x, t_tic163_min_norm=t_tic163_min_norm: x[29] - t_tic163_min_norm},  # t_tic163 > t_tic163_min_norm
            {'type': 'ineq', 'fun': lambda x, t_tic163_max_norm=t_tic163_max_norm: t_tic163_max_norm - x[29]},  # t_tic163 < t_tic163_max_norm
            {'type': 'ineq', 'fun': lambda x, t_tic166_min_norm=t_tic166_min_norm: x[30] - t_tic166_min_norm},  # t_tic166 > t_tic166_min_norm
            {'type': 'ineq', 'fun': lambda x, t_tic166_max_norm=t_tic166_max_norm: t_tic166_max_norm - x[30]},  # t_tic166 < t_tic166_max_norm
            {'type': 'ineq', 'fun': lambda x, t_tic172_min_norm=t_tic172_min_norm: x[31] - t_tic172_min_norm},  # t_tic172 > t_tic172_min_norm
            {'type': 'ineq', 'fun': lambda x, t_tic172_max_norm=t_tic172_max_norm: t_tic172_max_norm - x[31]},  # t_tic172 < t_tic172_max_norm

            # Constraints berdasarkan prediksi model Ridge
            {'type': 'ineq', 'fun': lambda x, T_furnace_norm=T_furnace_norm: ridge_model.predict(x.reshape(1, -1))[0][0] - T_furnace_norm},  # T_furnace > T_furnace_norm
            {'type': 'ineq', 'fun': lambda x, ni_met_norm=ni_met_norm: ridge_model.predict(x.reshape(1, -1))[0][1] - ni_met_norm},    # Ni_met > ni_met_norm
            {'type': 'ineq', 'fun': lambda x, C_met_low_norm=C_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][2] - C_met_low_norm},  # C_met > C_met_low_norm
            {'type': 'ineq', 'fun': lambda x, Si_met_low_norm=Si_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][3] - Si_met_low_norm},  # Si_met > Si_met_low_norm
            {'type': 'ineq', 'fun': lambda x, Si_met_high_norm=Si_met_high_norm: Si_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][3]},  # Si_met < Si_met_high_norm
            {'type': 'ineq', 'fun': lambda x, fe_met_low_norm=fe_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][4] - fe_met_low_norm},  # Fe_met > fe_met_low_norm
            {'type': 'ineq', 'fun': lambda x, s_met_low_norm=s_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][5] - s_met_low_norm},  # S_met > s_met_low_norm
            {'type': 'ineq', 'fun': lambda x, s_met_high_norm=s_met_high_norm: s_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][5]},  # S_met < s_met_high_norm
            {'type': 'ineq', 'fun': lambda x, ni_slag_low_norm=ni_slag_low_norm: ridge_model.predict(x.reshape(1, -1))[0][6] - ni_slag_low_norm},  # Ni_slag > ni_slag_low_norm
            {'type': 'ineq', 'fun': lambda x, fe_slag_low_norm=fe_slag_low_norm: ridge_model.predict(x.reshape(1, -1))[0][7] - fe_slag_low_norm},  # Fe_slag > fe_slag_low_norm
            {'type': 'ineq', 'fun': lambda x, T_kalsin_low_norm=T_kalsin_low_norm: ridge_model.predict(x.reshape(1, -1))[0][8] - T_kalsin_low_norm},    # T_kalsin > T_kalsin_low_norm
            {'type': 'ineq', 'fun': lambda x, pic_161_low_norm=pic_161_low_norm: ridge_model.predict(x.reshape(1, -1))[0][9] - pic_161_low_norm},  # pic_161 > pic_161_low_norm
            {'type': 'ineq', 'fun': lambda x, loi_kalsin_high=loi_kalsin_high: loi_kalsin_high - ridge_model.predict(x.reshape(1, -1))[0][10]},  # loi_kalsin < loi_kalsin_high
            {'type': 'ineq', 'fun': lambda x, loi_kalsin_low=loi_kalsin_low: ridge_model.predict(x.reshape(1, -1))[0][10] - loi_kalsin_low },  # loi_kalsin > loi_kalsin_low
        ]

        # Melakukan optimisasi menggunakan metode SLSQP dengan constraints yang didefinisikan
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            constraints=constraints_sample,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )

        # Memeriksa apakah optimisasi berhasil
        if result.success:
            optimized_x = result.x
            optimized_y = ridge_model.predict(optimized_x.reshape(1, -1))[0]

            # Menyimpan input dan output yang dioptimalkan
            optimized_x_list.append(optimized_x)
            optimized_y_list.append(optimized_y)
        else:
            print(f"Optimisasi gagal untuk sampel {i+1}: {result.message}")
            # Menambahkan NaN jika optimisasi gagal
            optimized_x_list.append([np.nan] * len(input_cols))
            optimized_y_list.append([np.nan] * len(output_cols))

    # Mengubah list input yang dioptimalkan menjadi DataFrame
    optimized_x_df = pd.DataFrame(optimized_x_list, columns=input_cols)

    # Inverse transform input yang dioptimalkan ke skala asli
    optimized_x_df_original = pd.DataFrame(
        scaler_x.inverse_transform(optimized_x_df.fillna(0)),
        columns=input_cols
    )

    # Mengubah list output yang dioptimalkan menjadi DataFrame
    optimized_y_df = pd.DataFrame(optimized_y_list, columns=output_cols)

    # Inverse transform output yang dioptimalkan ke skala asli
    optimized_y_df_original = pd.DataFrame(
        scaler_y.inverse_transform(optimized_y_df.fillna(0)),
        columns=output_cols
    )
    # print(optimized_x_df_original["voltage"])
    # print(optimized_x_df_original["current"])
    # print(optimized_x_df_original["load"])
    # print(optimized_x_df_original["rpm"])
    # print(optimized_x_df_original["pry_p"])
    # print(optimized_x_df_original["sec_p"])
    # print(optimized_x_df_original["pry_v"])
    # print(optimized_x_df_original["sec_v"])
    # print(optimized_x_df_original["total_coal"])
    # print(optimized_x_df_original["a_f_ratio"])
    # print(optimized_x_df_original["kg_tco"])
    # print(optimized_x_df_original["reductor_ratio"])
    # print(optimized_x_df_original["reductor_consume"])
    # print(optimized_x_df_original["t_tic162"])
    # print(optimized_x_df_original["t_tic163"])

    # Mengambil nilai dari DataFrame dan menyusunnya dalam dictionary
    output_y = {
        "furnace_temp": optimized_y_df_original["furnace_temp"].values[0],
        "ni_met": optimized_y_df_original["ni_met"].values[0],
        "c_met": optimized_y_df_original["c_met"].values[0],
        "si_met": optimized_y_df_original["si_met"].values[0],
        "fe_met": optimized_y_df_original["fe_met"].values[0],
        "s_met": optimized_y_df_original["s_met"].values[0],
        "ni_slag": optimized_y_df_original["ni_slag"].values[0],
        "fe_slag": optimized_y_df_original["fe_slag"].values[0],
        "t_kalsin": optimized_y_df_original["t_kalsin"].values[0],
        "pic_161": optimized_y_df_original["pic_161"].values[0],
        "loi_kalsin": optimized_y_df_original["loi_kalsin"].values[0]
    }



    # Mengambil nilai dari DataFrame dan menyusunnya dalam dictionary
    output = {
        "Tegangan (V)": optimized_x_df_original["voltage"].values[0],
        "Arus (A)": optimized_x_df_original["current"].values[0],
        "Load (MW)": optimized_x_df_original["load"].values[0],
        "rpm": optimized_x_df_original["rpm"].values[0],
        "pry_p": optimized_x_df_original["pry_p"].values[0],
        "sec_p": optimized_x_df_original["sec_p"].values[0],
        "pry_v": optimized_x_df_original["pry_v"].values[0],
        "sec_v": optimized_x_df_original["sec_v"].values[0],
        "total_coal": optimized_x_df_original["total_coal"].values[0],
        "a_f_ratio": optimized_x_df_original["a_f_ratio"].values[0],
        # "kg_tco": optimized_x_df_original["kg_tco"].values[0],
        "reductor_ratio": optimized_x_df_original["reductor_ratio"].values[0],
        "reductor_consume": optimized_x_df_original["reductor_consume"].values[0],
        "t_tic162": optimized_x_df_original["t_tic162"].values[0],
        "t_tic163": optimized_x_df_original["t_tic163"].values[0],
        "t_tic166": optimized_x_df_original["t_tic166"].values[0],
        "t_tic172": optimized_x_df_original["t_tic172"].values[0]
    }

    # Menyusun output menjadi tabel HTML untuk output pertama (output)
    output_html_1 = "<table style='width:100%; border: 1px solid black; border-collapse: collapse;'>"
    output_html_1 += "<tr><th style='padding: 8px; text-align: left;'>Parameter</th><th style='padding: 8px; text-align: left;'>Value</th></tr>"

    for param, value in output.items():
        output_html_1 += f"<tr><td style='padding: 8px; border: 1px solid black;'>{param}</td><td style='padding: 8px; border: 1px solid black;'>{value}</td></tr>"

    output_html_1 += "</table>"

    # Menyusun output menjadi tabel HTML untuk output kedua (output_y)
    output_html_2 = "<table style='width:100%; border: 1px solid black; border-collapse: collapse;'>"
    output_html_2 += "<tr><th style='padding: 8px; text-align: left;'>Parameter</th><th style='padding: 8px; text-align: left;'>Value</th></tr>"

    for param, value in output_y.items():
        output_html_2 += f"<tr><td style='padding: 8px; border: 1px solid black;'>{param}</td><td style='padding: 8px; border: 1px solid black;'>{value}</td></tr>"

    output_html_2 += "</table>"

    # Menampilkan kedua tabel HTML di Streamlit
    st.markdown("### Input", unsafe_allow_html=True)
    st.markdown(output_html_1, unsafe_allow_html=True)

    st.markdown("### Output Target", unsafe_allow_html=True)
    st.markdown(output_html_2, unsafe_allow_html=True)
